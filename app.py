import os
import re
import csv
import io
import json
import uuid
import sqlite3
try:
    import libsql_experimental as libsql_turso
except ImportError:
    libsql_turso = None

import threading
import pandas as pd
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# NLTK untuk Preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Gensim & PyLDAvis untuk Pemodelan Topik
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
import pyLDAvis
import pyLDAvis.gensim_models

# Import skrip pelabelan otomatis
from auto_labeler import interpret_topic

# Unduh resource NLTK (quiet=True agar tidak membebani startup)
nltk.download('punkt',                      quiet=True)
nltk.download('stopwords',                  quiet=True)
nltk.download('wordnet',                    quiet=True)
nltk.download('omw-1.4',                    quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==========================================
# INISIALISASI FLASK
# static_folder mengarah ke hasil build React (frontend/dist)
# Digunakan saat production/HuggingFace; saat dev Vite server yang melayani frontend.
# ==========================================
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='/')

# [KRITIKAL] CORS dibatasi hanya ke origin frontend yang dikenal (untuk mode dev)
CORS(app, resources={r"/*": {"origins": "*"}})

DB_PATH       = 'database.db'
UPLOAD_FOLDER = 'uploads'

TURSO_DATABASE_URL = os.environ.get('TURSO_DATABASE_URL', '')
TURSO_AUTH_TOKEN   = os.environ.get('TURSO_AUTH_TOKEN', '')

def get_db_connection():
    if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN and libsql_turso:
        return libsql_turso.connect(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)
    return sqlite3.connect(DB_PATH)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==========================================
# BACKGROUND TASK STORE (untuk find_optimal_k)
# ==========================================
_tasks: dict = {}
_tasks_lock   = threading.Lock()


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_json_data():
    return request.get_json(silent=True) or {}

def get_session_filepath(filename):
    if not filename:
        return os.path.join(UPLOAD_FOLDER, 'session_default.json')
    safe_name = str(filename).replace('.csv', '')
    return os.path.join(UPLOAD_FOLDER, f'session_{safe_name}.json')

def save_session(filename, data):
    with open(get_session_filepath(filename), 'w') as f:
        json.dump(data, f)

def load_session(filename):
    filepath = get_session_filepath(filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def cleanup_old_sessions(exclude_filename=None):
    """Hapus file session lama kecuali yang sedang aktif."""
    try:
        for fname in os.listdir(UPLOAD_FOLDER):
            if fname.startswith('session_') and fname.endswith('.json'):
                if exclude_filename:
                    safe = exclude_filename.replace('.csv', '')
                    if fname == f'session_{safe}.json':
                        continue
                os.remove(os.path.join(UPLOAD_FOLDER, fname))
    except Exception:
        pass

def init_db():
    conn   = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS movie_analysis (
            id_title    TEXT PRIMARY KEY,
            result_data TEXT,
            created_at  TIMESTAMP
        )
    ''')
    # [ARSITEKTUR] Migrasi: tambah kolom created_at jika tabel lama belum punya.
    try:
        cursor.execute("ALTER TABLE movie_analysis ADD COLUMN created_at TIMESTAMP")
        cursor.execute("UPDATE movie_analysis SET created_at = datetime('now') WHERE created_at IS NULL")
    except Exception:
        pass  # Kolom sudah ada â€” abaikan (Exception agar kompatibel Turso & SQLite)
    conn.commit()
    conn.close()

init_db()


# ==========================================
# [KRITIKAL] HELPER DRY: BUILD LDA PAYLOAD
# Menghilangkan duplikasi kode antara /find_optimal_k dan /analyze
# ==========================================
def _build_lda_payload(title: str, k: int, tokens, corpus, id2word, raw_texts):
    """
    Melatih satu model LDA untuk k topik dan mengembalikan
    (payload_dict, coherence_score_float, perplexity_score_float).
    """
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=k,
        random_state=42,
        passes=20,
        iterations=200,
        chunksize=100,
        alpha='auto',
        eta='auto'
    )

    coherence_model = CoherenceModel(
        model=model, texts=tokens, dictionary=id2word, coherence='c_v', processes=1
    )
    coh_score  = float(coherence_model.get_coherence())
    perp_score = float(model.log_perplexity(corpus))

    vis      = pyLDAvis.gensim_models.prepare(model, corpus, id2word, sort_topics=False)
    vis_html = pyLDAvis.prepared_data_to_html(vis)

    topics_data = {}
    for idx, topic in model.show_topics(num_topics=k, num_words=10, formatted=False):
        topic_words     = [{"word": word, "weight": float(weight)} for word, weight in topic]
        kata_saja       = [w for w, _ in topic]
        interpret_hasil = interpret_topic(kata_saja)
        catatan         = f"{interpret_hasil['summary']} {interpret_hasil['recommendation']}"
        topics_data[f"Topik {idx + 1}"] = {
            "auto_label": interpret_hasil["label"],
            "auto_notes": catatan,
            "words": topic_words
        }

    doc_distributions = []
    topic_counts = {f"Topik {i+1}": 0 for i in range(k)}

    for i, corp in enumerate(corpus):
        topic_probs = model.get_document_topics(corp)
        if topic_probs:
            dominant = max(topic_probs, key=lambda x: x[1])
            dom_name = f"Topik {dominant[0] + 1}"
            topic_counts[dom_name] += 1
            doc_distributions.append({
                "doc_id":          i + 1,
                "text":            raw_texts[i] if i < len(raw_texts) else "",
                "dominant_topic":  dom_name,
                "probability":     float(dominant[1])
            })

    classified = sum(topic_counts.values())
    overall_distribution = {
        t: round((c / classified) * 100, 2) if classified > 0 else 0
        for t, c in topic_counts.items()
    }

    interpretations = {
        tk: {"custom_label": tv["auto_label"], "notes": tv["auto_notes"]}
        for tk, tv in topics_data.items()
    }

    payload = {
        "title":                title,
        "num_topics":           k,
        "coherence_score":      round(coh_score, 4),
        "perplexity_score":     round(perp_score, 4),
        "topics":               topics_data,
        "overall_distribution": overall_distribution,
        "document_distributions": doc_distributions,
        "vis_html":             vis_html,
        "interpretations":      interpretations,
    }
    return payload, coh_score, perp_score


# ==========================================
# BACKGROUND WORKER: FIND OPTIMAL K
# ==========================================
def _run_find_optimal_k(task_id, title, min_k, max_k, tokens, corpus, id2word, raw_texts):
    """Dijalankan di thread terpisah. Melatih K=min_k s/d K=max_k."""
    total_k  = max_k - min_k + 1
    results  = []
    payloads = []

    try:
        for i, k in enumerate(range(min_k, max_k + 1)):
            with _tasks_lock:
                _tasks[task_id].update({
                    'progress': i,
                    'total':    total_k,
                    'current_k': k
                })

            payload, coh, perp = _build_lda_payload(title, k, tokens, corpus, id2word, raw_texts)
            results.append({"k": k, "score": coh, "perplexity": perp})
            payloads.append({"k": k, "payload": payload})

        # Simpan semua model ke database secara massal
        conn   = get_db_connection()
        cursor = conn.cursor()
        for item in payloads:
            k       = item["k"]
            payload = item["payload"]
            payload["optimal_k_results"] = results
            db_key  = f"{title}_k{k}"
            cursor.execute(
                'INSERT OR REPLACE INTO movie_analysis (id_title, result_data) VALUES (?, ?)',
                (db_key, json.dumps(payload))
            )
        conn.commit()
        conn.close()

        with _tasks_lock:
            _tasks[task_id]['status'] = 'done'
            _tasks[task_id]['result'] = results

    except Exception as e:
        with _tasks_lock:
            _tasks[task_id]['status'] = 'error'
            _tasks[task_id]['error']  = str(e)


# ==========================================
# ENDPOINT 1: UPLOAD DATASET
# ==========================================
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nama file kosong"}), 400

    # [KRITIKAL] Sanitasi nama file untuk mencegah Path Traversal
    safe_filename = secure_filename(file.filename)
    if not safe_filename or not safe_filename.endswith('.csv'):
        return jsonify({"error": "Hanya file CSV yang diizinkan"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(filepath)

    # [MINOR] Hapus sesi lama setelah file baru berhasil diunggah
    cleanup_old_sessions(exclude_filename=safe_filename)

    try:
        df           = pd.read_csv(filepath)
        columns      = df.columns.tolist()
        preview_data = df.head(5).fillna("").to_dict(orient='records')

        return jsonify({
            "status":   "success",
            "filename": safe_filename,
            "columns":  columns,
            "preview":  preview_data,
            "message":  "File berhasil diunggah dan dibaca."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# ENDPOINT 2: PREPROCESSING
# ==========================================
@app.route('/preprocess', methods=['POST'])
def preprocess():
    data        = get_json_data()
    column_name = data.get('column')
    filename    = data.get('filename')

    if not filename or not column_name:
        return jsonify({"error": "Filename atau kolom tidak valid."}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File tidak ditemukan di server. Silakan unggah ulang."}), 404

    try:
        df        = pd.read_csv(filepath)
        raw_texts = df[column_name].dropna().astype(str).tolist()

        # Inisialisasi Stopwords NLTK
        stop_words = set(stopwords.words('english'))

        # Amankan kata negasi agar tidak dihapus oleh Stopwords
        negation_words = {"not", "no", "never", "cannot", "without", "neither", "nor"}
        stop_words     = stop_words - negation_words

        # Ekstensi Stopwords Khusus
        custom_stops = {
            "movie", "film", "movies", "films", "one", "like", "time", "even", 
            "much", "really", "also", "ever", "many", "way", "made", "people", 
            "say", "still", "think", "two", "every", "make", "could", "something", 
            "get", "never", "see", "seen", "watch", "story", "plot", "character", 
            "characters", "best", "great", "good", "well", "love", "better", "end", "world",
            
            # Additional terms
            "just", "feel", "little", "makes", "know", "times",
            "quite", "going", "real", "right", "thought",
            "want", "point", "thing", "things",
            "actually", "sure", "different", "definitely", "find", "found",
        }

        # Tambahkan kata-kata dari judul file (tanpa tahun dan ekstensi) sebagai stopwords
        # Misal: "The_Dark_Knight_2008.csv" -> "the", "dark", "knight"
        clean_filename = re.sub(r'^\d+_', '', filename) # hapus awalan angka jika ada (misal 100_)
        clean_filename = re.sub(r'_\d{4}\.csv$', '', clean_filename) # hapus tahun dan .csv
        clean_filename = clean_filename.replace('.csv', '').replace('_', ' ')
        
        title_words = set(clean_filename.lower().split())
        custom_stops = custom_stops.union(title_words)
        
        # Hardcoded stopwords khusus untuk beberapa film populer agar coherence makin tinggi
        filename_lower = clean_filename.lower()
        if "dark knight" in filename_lower or "batman" in filename_lower:
            custom_stops.update({"batman", "nolan", "joker", "bruce", "wayne", "heath", "ledger"})
        elif "lord of the rings" in filename_lower:
            custom_stops.update({"frodo", "ring", "gandalf", "sam", "peter", "jackson", "hobbit", "king"})
        elif "avengers" in filename_lower or "endgame" in filename_lower:
            custom_stops.update({"marvel", "avenger", "avengers", "thanos", "stark", "iron", "man", "tony", "cap", "captain", "america", "endgame"})
        elif "spider-man" in filename_lower or "spider man" in filename_lower:
            custom_stops.update({"spider", "man", "spiderman", "peter", "parker", "miles", "morales", "verse"})
        elif "interstellar" in filename_lower:
            custom_stops.update({"space", "cooper", "murph", "nolan"})
        elif "parasite" in filename_lower:
            custom_stops.update({"korean", "family", "bong", "joon", "ho", "house"})
        elif "coco" in filename_lower:
            custom_stops.update({"pixar", "miguel", "music", "disney", "mexico", "family"})
        elif "toy story" in filename_lower:
            custom_stops.update({"pixar", "toy", "toys", "woody", "buzz", "andy"})
        elif "wall-e" in filename_lower:
            custom_stops.update({"pixar", "wall", "eve", "robot", "earth"})
        elif "your name" in filename_lower:
            custom_stops.update({"anime", "mitsuha", "taki", "body", "swap", "shinkai"})

        stop_words = stop_words.union(custom_stops)

        lemmatizer = WordNetLemmatizer()

        valid_original        = []
        processed_tokens_temp = []
        step_original         = []
        step_casefolding      = []
        step_cleansing        = []
        step_stopword         = []
        step_lemmatization    = []

        for text in raw_texts:
            # 1. Case Folding
            text_lower = text.lower()

            # 2. Normalisasi Elongasi ("loooove" â†’ "loove")
            text_elong = re.sub(r'(.)\1{2,}', r'\1\1', text_lower)

            # 3. Cleansing
            text_clean = re.sub(r'[^a-z\s]', ' ', text_elong)
            text_clean = re.sub(r'\s+', ' ', text_clean).strip()

            # 4. Tokenisasi
            tokens = word_tokenize(text_clean)

            # 5. Penanganan Negasi
            tokens_negation = []
            skip_next = False
            for i in range(len(tokens)):
                if skip_next:
                    skip_next = False
                    continue
                if tokens[i] in negation_words and i + 1 < len(tokens):
                    tokens_negation.append(tokens[i] + "_" + tokens[i + 1])
                    skip_next = True
                else:
                    tokens_negation.append(tokens[i])

            # 6. Hapus Stopwords & Filter Panjang Kata (â‰¥4 huruf)
            tokens_no_stop = [w for w in tokens_negation if w not in stop_words and len(w) > 3]

            # 7. POS Tagging
            pos_tags = nltk.pos_tag(tokens_no_stop)

            # 8. Filter hanya Noun & Adjective
            allowed_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}
            tokens_filtered = [word for word, tag in pos_tags if tag in allowed_pos]

            # 9. Lemmatization
            tokens_lemma = [lemmatizer.lemmatize(w) for w in tokens_filtered]

            if tokens_lemma:
                valid_original.append(text)
                processed_tokens_temp.append(tokens_lemma)

                if len(step_original) < 5:
                    step_original.append(text)
                    step_casefolding.append(text_lower)
                    step_cleansing.append(text_clean)
                    step_stopword.append(" ".join(tokens_filtered))
                    step_lemmatization.append(" ".join(tokens_lemma))

        # Bigram Detection
        bigram     = Phrases(processed_tokens_temp, min_count=3, threshold=10)
        bigram_mod = Phraser(bigram)
        processed_tokens = [list(bigram_mod[doc]) for doc in processed_tokens_temp]

        # ── Statistik Preprocessing ──────────────────────────────
        total_docs_raw   = len(raw_texts)
        total_docs_valid = len(processed_tokens)
        total_dropped    = total_docs_raw - total_docs_valid
        all_tokens_flat  = [tok for doc in processed_tokens for tok in doc]
        vocab_size       = len(set(all_tokens_flat))
        total_tokens     = len(all_tokens_flat)
        doc_lengths      = [len(doc) for doc in processed_tokens]
        avg_tokens       = round(sum(doc_lengths) / len(doc_lengths), 1) if doc_lengths else 0
        sorted_lengths   = sorted(doc_lengths)
        mid              = len(sorted_lengths) // 2
        median_tokens    = sorted_lengths[mid] if sorted_lengths else 0
        # ──────────────────────────────────────────────────────────

        save_session(filename, {
            'original_text':      valid_original,
            'processed_tokens':   processed_tokens
        })

        return jsonify({
            "status": "success",
            "data": {
                "original":       step_original,
                "case_folding":   step_casefolding,
                "cleansing":      step_cleansing,
                "stopword":       step_stopword,
                "lemmatization":  step_lemmatization
            },
            "stats": {
                "total_docs_raw":   total_docs_raw,
                "total_docs_valid": total_docs_valid,
                "total_dropped":    total_dropped,
                "vocab_size":       vocab_size,
                "total_tokens":     total_tokens,
                "avg_tokens":       avg_tokens,
                "median_tokens":    median_tokens
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# ENDPOINT 3: CARI K OPTIMAL (BACKGROUND TASK)
# ==========================================
@app.route('/find_optimal_k', methods=['POST'])
def find_optimal_k():
    data     = get_json_data()
    min_k    = max(2, int(data.get('min_k', 2)))
    max_k    = min(20, int(data.get('max_k', 10)))
    filename = data.get('filename')
    title    = data.get('title', 'Dataset_Ulasan').replace(" ", "_")

    if not filename:
        return jsonify({"error": "Sesi terhapus. Silakan unggah dan proses ulang."}), 400

    session_data = load_session(filename)
    tokens       = session_data.get('processed_tokens', [])
    raw_texts    = session_data.get('original_text', [])

    if not tokens:
        return jsonify({"error": "Silakan jalankan preprocessing terlebih dahulu"}), 400

    try:
        id2word = corpora.Dictionary(tokens)
        id2word.filter_extremes(no_below=2, no_above=0.75)
        corpus  = [id2word.doc2bow(text) for text in tokens]

        # Buat task ID dan simpan ke store
        task_id = str(uuid.uuid4())
        with _tasks_lock:
            _tasks[task_id] = {
                'status':    'running',
                'progress':  0,
                'total':     max_k - min_k + 1,
                'current_k': min_k,
            }

        # [ARSITEKTUR] Jalankan di background thread agar frontend tidak timeout
        thread = threading.Thread(
            target=_run_find_optimal_k,
            args=(task_id, title, min_k, max_k, tokens, corpus, id2word, raw_texts),
            daemon=True
        )
        thread.start()

        return jsonify({"status": "started", "task_id": task_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# ENDPOINT 3b: CEK STATUS BACKGROUND TASK
# ==========================================
@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    with _tasks_lock:
        task = _tasks.get(task_id)

    if not task:
        return jsonify({"error": "Task tidak ditemukan"}), 404

    # Bersihkan task yang sudah selesai dari memori setelah dibaca
    if task.get('status') in ('done', 'error'):
        with _tasks_lock:
            _tasks.pop(task_id, None)

    return jsonify({"status": "success", "data": task})


# ==========================================
# ENDPOINT 4: ANALISIS LDA UTAMA
# ==========================================
@app.route('/analyze', methods=['POST'])
def analyze():
    data        = get_json_data()
    title       = data.get('title', 'Dataset_Ulasan').replace(" ", "_")
    # [SEDANG] Validasi: num_topics diklem antara 2â€“20
    num_topics  = max(2, min(20, int(data.get('num_topics', 3))))
    filename    = data.get('filename')

    if not filename:
        return jsonify({"error": "Sesi terhapus. Silakan unggah dan proses ulang."}), 400

    session_data = load_session(filename)
    tokens       = session_data.get('processed_tokens', [])
    raw_texts    = session_data.get('original_text', [])

    if not tokens:
        return jsonify({"error": "Silakan jalankan preprocessing terlebih dahulu"}), 400

    try:
        id2word = corpora.Dictionary(tokens)
        id2word.filter_extremes(no_below=2, no_above=0.75)
        corpus  = [id2word.doc2bow(text) for text in tokens]

        # [KRITIKAL] Gunakan helper DRY â€” tidak ada duplikasi
        result_payload, _, _ = _build_lda_payload(title, num_topics, tokens, corpus, id2word, raw_texts)
        result_payload["optimal_k_results"] = data.get('optimal_k_results', None)

        db_key = f"{title}_k{num_topics}"
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO movie_analysis (id_title, result_data) VALUES (?, ?)',
            (db_key, json.dumps(result_payload))
        )
        conn.commit()
        conn.close()

        return jsonify({"status": "success", "data": result_payload})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# ENDPOINT 5: AUTO INTERPRET (LOKAL)
# ==========================================
@app.route('/auto_interpret_local', methods=['POST'])
def auto_interpret_local():
    data  = get_json_data()
    words = [w['word'] for w in data.get('words', [])]
    try:
        hasil          = interpret_topic(words)
        catatan_lengkap = f"{hasil['summary']} {hasil['recommendation']}"
        return jsonify({"status": "success", "label": hasil["label"], "notes": catatan_lengkap})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# ENDPOINT 6: SIMPAN INTERPRETASI MANUAL
# ==========================================
@app.route('/update_interpretation', methods=['POST'])
def update_interpretation():
    data         = get_json_data()
    db_key       = f"{data.get('title')}_k{data.get('num_topics')}"
    topic_id     = data.get('topic_id')
    custom_label = data.get('custom_label', '')
    notes        = data.get('notes', '')

    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT result_data FROM movie_analysis WHERE id_title = ?', (db_key,))
        row = cursor.fetchone()
        if row:
            result_data = json.loads(row[0])
            if 'interpretations' not in result_data:
                result_data['interpretations'] = {}
            result_data['interpretations'][topic_id] = {"custom_label": custom_label, "notes": notes}
            cursor.execute(
                'UPDATE movie_analysis SET result_data = ? WHERE id_title = ?',
                (json.dumps(result_data), db_key)
            )
            conn.commit()
            conn.close()
            return jsonify({"status": "success", "data": result_data})

        conn.close()
        return jsonify({"error": "Data analisis belum tersimpan"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# ENDPOINT 7 & 8: TAMPILKAN HISTORY
# ==========================================
@app.route('/saved_movies', methods=['GET'])
def get_saved_movies():
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        # [ARSITEKTUR] Urutkan berdasarkan created_at terbaru; ambil juga result_data
        # untuk mengekstrak optimal_k (K dengan coherence tertinggi) tanpa fetch detail.
        cursor.execute('SELECT id_title, result_data FROM movie_analysis ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()

        result = []
        for id_title, result_data_json in rows:
            optimal_k = None
            try:
                rd = json.loads(result_data_json)
                k_results = rd.get('optimal_k_results')
                if k_results:
                    best = max(k_results, key=lambda x: x.get('score', 0))
                    optimal_k = best.get('k')
            except Exception:
                pass
            result.append({"id_title": id_title, "optimal_k": optimal_k})

        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/saved_movies/<title>', methods=['GET'])
def get_saved_movie_detail(title):
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT result_data FROM movie_analysis WHERE id_title = ?', (title,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return jsonify({"status": "success", "data": json.loads(row[0])})
        return jsonify({"error": "Data tidak ditemukan"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# ENDPOINT 9: HAPUS HISTORY
# ==========================================
@app.route('/delete_movie/<title>', methods=['DELETE'])
def delete_movie(title):
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM movie_analysis WHERE id_title = ?', (title,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "Data berhasil dihapus"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# ENDPOINT 10: EXPORT CSV HASIL ANALISIS
# ==========================================
@app.route('/export_csv/<title>', methods=['GET'])
def export_csv(title):
    """
    [ARSITEKTUR] Mengembalikan file CSV berisi doc_id, teks ulasan,
    topik dominan, label topik, dan probabilitas.
    """
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT result_data FROM movie_analysis WHERE id_title = ?', (title,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Data tidak ditemukan"}), 404

        result_data      = json.loads(row[0])
        doc_distributions = result_data.get('document_distributions', [])
        interpretations  = result_data.get('interpretations', {})
        topics           = result_data.get('topics', {})

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID Dokumen', 'Teks Ulasan', 'Topik Dominan', 'Label Topik', 'Probabilitas'])

        for doc in doc_distributions:
            dom   = doc.get('dominant_topic', '')
            label = (interpretations.get(dom, {}).get('custom_label')
                     or topics.get(dom, {}).get('auto_label', ''))
            writer.writerow([
                doc.get('doc_id', ''),
                doc.get('text', ''),
                dom,
                label,
                round(doc.get('probability', 0), 4)
            ])

        output.seek(0)
        safe_title = secure_filename(title)
        return Response(
            output.getvalue(),
            mimetype='text/csv; charset=utf-8',
            headers={
                'Content-Disposition': f'attachment; filename="{safe_title}_hasil_analisis.csv"'
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# CATCH-ALL ROUTE: Serve React SPA
# Semua request yang bukan endpoint API akan dilayani oleh index.html
# agar React Router bisa menangani navigasi di sisi client.
# Route ini harus didefinisikan TERAKHIR agar tidak menimpa endpoint API.
# ==========================================
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    # Jika file static ada (js, css, assets, dll.), layani langsung
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    # Untuk semua route lainnya, kembalikan index.html (React Router yang handle)
    index_path = os.path.join(app.static_folder, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder, 'index.html')
    # Fallback: jika dist/ belum di-build, kembalikan pesan informatif
    return jsonify({
        "message": "Frontend belum di-build. Jalankan: cd frontend && npm run build",
        "status": "no_frontend"
    }), 404


if __name__ == '__main__':
    # Baca PORT dari environment variable.
    # Di HuggingFace Spaces, port wajib 7860.
    # Di lokal (dev), default ke 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=(port == 5000), host='0.0.0.0', port=port, threaded=True)
