import sqlite3

DATABASE_NAME = "users.db"

def clear_database(delete_all=True):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()

    if delete_all:
        # Clear all data but keep the table
        c.execute("DELETE FROM users")
        print("All records deleted.")
    else:
        # Example: Delete specific record (e.g., based on email)
        email_to_delete = "example@example.com"
        c.execute("DELETE FROM users WHERE email = ?", (email_to_delete,))
        print(f"Records for {email_to_delete} deleted.")

    conn.commit()
    conn.close()

# Call the function to clear data
clear_database(delete_all=True)
