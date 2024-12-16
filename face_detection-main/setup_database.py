import sqlite3

# Connect to SQLite database (creates the file if it doesn't exist)
conn = sqlite3.connect("tasks.db")

# Create a cursor object
cursor = conn.cursor()

# Create the "tasks" table
cursor.execute("""
CREATE TABLE IF NOT EXISTS tasks (
    worker_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    schedule TEXT NOT NULL,
    tasks TEXT NOT NULL
)
""")

# Insert sample data
sample_data = [
    (1, 'John Doe', 'Shift 1: 9:00 AM - 5:00 PM', 'Inspect machinery, Safety drills'),
    (2, 'Jane Smith', 'Shift 2: 2:00 PM - 10:00 PM', 'Assemble parts, Review checklist'),
    (3, 'Robert Brown', 'Shift 1: 9:00 AM - 5:00 PM', 'Test samples, Report progress')
]

cursor.executemany("INSERT INTO tasks (worker_id, name, schedule, tasks) VALUES (?, ?, ?, ?)", sample_data)

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database setup completed. Sample data inserted.")
