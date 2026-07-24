from typing import Any

# ==============================================================================
# 1. KAMUS TEMA UTAMA (Dalam bentuk dasar/lemma)
# ==============================================================================
TOPIC_THEMES = [
    {
        "label": "Animasi, Visual, dan Sinematografi",
        "keywords": {
            "animation", "animated", "pixar", "beautiful", "effect", "screen", 
            "look", "visual", "art", "style", "graphic", "color", "cgi",
            "walle", "toy", "spiderman", "coco"
        },
        "focus": "tampilan grafis, gaya animasi, dan keindahan visual film",
        "recommendation": "Coba telusuri apakah efek visual ini membuat pengalaman menonton jadi lebih berkesan.",
    },
    {
        "label": "Karakter dan Akting",
        "keywords": {
            "character", "acting", "actor", "heath", "joker", 
            "batman", "cast", "performance", "villain", "hero", "buzz", "human",
            "role", "play"
        },
        "focus": "kualitas akting aktor dan seberapa menarik karakter yang dimainkan",
        "recommendation": "Fokuskan analisis pada tokoh mana yang aktingnya paling membekas di hati penonton.",
    },
    {
        "label": "Alur Cerita dan Penyutradaraan",
        "keywords": {
            "plot", "scene", "ending", "director", "nolan", "cinema", 
            "trilogy", "book", "original", "story", "writing", "script",
            "twist", "narrative", "part", "second", "last"
        },
        "focus": "kualitas jalan cerita, naskah, dan cara sutradara mengemas film",
        "recommendation": "Lihat apakah penonton merasa alur ceritanya masuk akal dan memuaskan.",
    },
    {
        "label": "Aksi dan Dunia Fiksi (Sci-Fi)",
        "keywords": {
            "world", "space", "interstellar", "earth", "epic", "king", "return", 
            "action", "universe", "time", "fight", "war", "ring", "sci-fi"
        },
        "focus": "adegan aksi, efek fiksi ilmiah, dan seberapa epik dunia yang dibangun",
        "recommendation": "Amati reaksi penonton terhadap serunya adegan laga atau uniknya dunia fiksi di film ini.",
    },
    {
        "label": "Emosi, Pesan Moral, dan Musik",
        "keywords": {
            "love", "life", "family", "emotional", "heart", "felt", "music", 
            "score", "soundtrack", "cry", "tear", "sad", "touching", "message"
        },
        "focus": "pesan moral, seberapa menyentuh ceritanya, serta kualitas musik pengiring",
        "recommendation": "Perhatikan apakah penonton merasa terharu atau terbawa suasana berkat cerita dan musiknya.",
    },
]

# ==============================================================================
# 2. ATURAN ASPEK PENDUKUNG (SENTIMEN)
# ==============================================================================
ASPECT_RULES = [
    {
        "name": "Pujian Mahakarya (Masterpiece)",
        "keywords": {"masterpiece", "perfect", "amazing", "truly", "best", "epic", "brilliant", "great", "excellent"},
        "signal": "pujian yang sangat tinggi",
        "recommendation": "Gali ulasan lebih lanjut untuk mencari tahu apa alasan utama yang membuat film ini dianggap sangat sempurna."
    },
    {
        "name": "Dampak Emosional Mendalam",
        "keywords": {"emotional", "heart", "felt", "cry", "love", "touching", "sad"},
        "signal": "perasaan haru yang kuat",
        "recommendation": "Cari tahu adegan atau pesan apa yang paling sukses membuat penonton sedih atau tersentuh."
    },
    {
        "name": "Ikonografi Sutradara/Studio",
        "keywords": {"pixar", "nolan", "disney", "marvel"},
        "signal": "harapan tinggi pada studio pembuatnya",
        "recommendation": "Bandingkan komentar penonton dengan reputasi studio atau film-film mereka sebelumnya."
    },
    {
        "name": "Elemen Kelam dan Ketegangan",
        "keywords": {"dark", "action", "joker", "tension", "scary", "thriller", "suspense"},
        "signal": "nuansa film yang tegang atau gelap",
        "recommendation": "Periksa apakah suasana tegang atau gelap ini justru menjadi daya tarik utama bagi penonton."
    },
    {
        "name": "Kritik atau Kekecewaan",
        "keywords": {"boring", "bad", "worst", "terrible", "waste", "disappointing", "stupid"},
        "signal": "rasa kecewa dari penonton",
        "recommendation": "Identifikasi keluhan utama penonton, apakah karena jalan cerita yang membosankan atau faktor lainnya."
    }
]

# ==============================================================================
# 3. ENGINE PELABELAN
# ==============================================================================

def _humanize_words(words: list[str], limit: int = 4) -> str:
    selected = [word.replace("_", " ") for word in words[:limit]]
    if not selected: return "kata-kata tersebut"
    if len(selected) == 1: return f'"{selected[0]}"'
    return ", ".join(f'"{w}"' for w in selected[:-1]) + f', dan "{selected[-1]}"'

def _weighted_matches(words: list[str], keywords: set[str]) -> tuple[float, list[str]]:
    matches = sorted(set(words) & keywords)
    score = sum(1.0 / (i + 1) for i, w in enumerate(words) if w in keywords)
    return score + (len(matches) * 0.15), matches

def interpret_topic(words: list[str]) -> dict[str, Any]:
    scored_themes = []
    for theme in TOPIC_THEMES:
        score, matches = _weighted_matches(words, theme["keywords"])
        if matches: scored_themes.append((score, theme))
    
    scored_aspects = []
    for aspect in ASPECT_RULES:
        score, matches = _weighted_matches(words, aspect["keywords"])
        if matches: scored_aspects.append((score, aspect))
    
    evidence = _humanize_words(words)
    
    if not scored_themes:
        return {
            "label": "Topik Umum",
            "summary": f"Topik ulasan ini cukup beragam. Namun jika melihat kemunculan kata seperti {evidence}, penonton menyoroti beberapa aspek umum dari film.",
            "recommendation": "Anda bisa meninjau ulang ulasan secara manual untuk menemukan maksud yang lebih spesifik."
        }

    best_theme = max(scored_themes, key=lambda x: x[0])[1]
    best_aspect = max(scored_aspects, key=lambda x: x[0])[1] if scored_aspects else None
    
    label = best_theme["label"]
    focus = best_theme["focus"]
    rec = best_aspect["recommendation"] if best_aspect else best_theme["recommendation"]
    
    summary = f"Topik ulasan ini sebagian besar berfokus pada {focus}. Hal ini terlihat dari seberapa sering penonton menyebutkan kata {evidence}."
    
    return {
        "label": label,
        "summary": summary,
        "recommendation": rec
    }