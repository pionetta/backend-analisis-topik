import sqlite3

DB_PATH = 'database.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("DELETE FROM movie_analysis WHERE id_title LIKE 'Dataset_Ulasan%'")
deleted = cursor.rowcount
conn.commit()

cursor.execute('SELECT id_title FROM movie_analysis ORDER BY created_at DESC')
remaining = cursor.fetchall()
conn.close()

print(f'Berhasil menghapus {deleted} record "Dataset_Ulasan" dari database.')
print(f'Sisa record: {len(remaining)}')
for r in remaining:
    print(' -', r[0])
