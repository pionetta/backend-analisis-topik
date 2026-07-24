import sqlite3

DB_PATH = 'database.db'

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Cek kolom yang ada
cursor.execute('PRAGMA table_info(movie_analysis)')
cols = [col[1] for col in cursor.fetchall()]
print(f"Kolom saat ini: {cols}")

if 'created_at' not in cols:
    # SQLite tidak mendukung DEFAULT CURRENT_TIMESTAMP via ALTER TABLE
    # Solusi: tambah kolom NULL dulu, lalu isi dengan waktu sekarang
    cursor.execute('ALTER TABLE movie_analysis ADD COLUMN created_at TIMESTAMP')
    cursor.execute("UPDATE movie_analysis SET created_at = datetime('now')")
    conn.commit()
    print("Kolom 'created_at' berhasil ditambahkan dan diisi!")
else:
    print("Kolom 'created_at' sudah ada, tidak perlu migrasi.")

# Verifikasi
cursor.execute('SELECT id_title, created_at FROM movie_analysis ORDER BY created_at DESC LIMIT 5')
rows = cursor.fetchall()
print(f"\nSample data ({len(rows)} baris):")
for r in rows:
    print(r)

conn.close()
print("\nMigrasi selesai!")
