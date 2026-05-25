import sqlite3

conn = sqlite3.connect("memory.db")
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("=== TABLES ===")
for t in tables:
    print(f"  - {t[0]}")

# Show all data in each table
for t in tables:
    table_name = t[0]
    print(f"\n=== TABLE: {table_name} ===")
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"Columns: {cols}")
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    print(f"Total rows: {len(rows)}")
    for i, row in enumerate(rows, 1):
        print(f"\n  Row {i}:")
        for col, val in zip(cols, row):
            print(f"    {col}: {val}")

conn.close()
print("\nDone.")
