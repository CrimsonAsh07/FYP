import sqlite3
import datetime

# Database connection and table creation (assuming table doesn't exist)
conn = sqlite3.connect('surveillance_db.db') # connect to sqlite
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS entries (
                camera_id INTEGER,
                person_id INTEGER,
                entry_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

# Function to record an entry
def record_entry(camera_id, person_id, conn):
  """
  Records an entry for a person entering a camera feed in the database.

  Args:
      camera_id: The ID of the camera feed where the person entered.
      person_id: The ID of the person who entered.
  """
  c=conn.cursor()  
  # Get current timestamp
  current_timestamp = datetime.datetime.now()

  # Insert data into the table
  c.execute("INSERT INTO entries (camera_id, person_id) VALUES (?, ?)", (camera_id, person_id))

  # Save changes to the database
  conn.commit()

# Example usage (replace with your camera feed processing logic)
if __name__ == "__main__":
    entries = c.execute("SELECT * from entries where camera_id = 1")
    if entries:
        print("Entry Records:")
        for row in entries:
            camera_id, person_id, entry_timestamp = row
            print(f"Camera ID: {camera_id}, Person ID: {person_id}, Entry Timestamp: {entry_timestamp}")
    else:
        print("No entries found in the 'entries' table.")

# Close the connection
conn.close()
  