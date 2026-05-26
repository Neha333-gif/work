#  1. AI Email Automation Agent
from groq import Groq
import os
import json
import sqlite3 as sql

# Initialize Groq client
# Use environment variable GROQ_API_KEY or provide your API key
api_key = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
client = Groq(api_key=api_key)

# In-memory store of processed emails for the current session
history = []

# Ensure database and table exist upon import
def sql_database():
    """Initializes the SQLite database."""
    conn = sql.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email_text TEXT,
        priority TEXT,
        category TEXT,
        reason TEXT,
        reply TEXT,
        followup TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

sql_database()

def store_email(email_text: str, priority: str, category: str, reason: str, reply: str, followup: str):
    """Stores processed email details in the SQLite database."""
    conn = sql.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO history (email_text, priority, category, reason, reply, followup)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (email_text, priority, category, reason, reply, followup))
    conn.commit()
    conn.close()
    print("Email stored in database.")

def read_email(email_text: str) -> str:
    """Prints the email text to the console."""
    border = 30 * "="
    print(border)
    print(email_text)
    print(border)
    return email_text

def process_email_full(email_text: str):
    """
    Performs classification, drafting, and follow-up scheduling in a single API call
    using structured JSON output for lowest possible latency.
    """
    prompt = f"""
    Analyze the following email and generate a response.
    
    Email content: {email_text}
    
    Rules:
    - Priority: 'high' if urgent/immediate, 'medium' if inquiry/response, 'low' otherwise.
    - Category: group into 'sales', 'technical', 'feedback', or 'general'.
    - Reason: brief explanation for the category/priority.
    - Reply: Draft a professional reply to the email.
    - Followup: Suggest a follow-up action plan. 
      CRITICAL INSTRUCTION: DO NOT use markdown formatting such as ** for bold or * for italics. 
      Use uppercase letters for section headers instead. The text must be clean plain text.
    
    Return a JSON object with EXACTLY these keys:
    "priority", "category", "reason", "reply", "followup"
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        data = json.loads(raw_text.strip())

        priority = data.get("priority", "low").strip().lower()
        category = data.get("category", "general").strip().lower()
        reason   = data.get("reason", "No reason provided")
        reply    = data.get("reply", "No reply generated")
        followup = data.get("followup", "No follow-up generated")

        entry = {
            "email_text": email_text,
            "priority": priority,
            "category": category,
            "reason": reason,
            "reply": reply,
            "followup": followup
        }
        history.append(entry)

        # Also store in the database
        store_email(email_text, priority, category, reason, reply, followup)

        print(f"\n[Groq] Processed email -> Priority: {priority.upper()} | Category: {category}")

        return (priority, category, reason, reply, followup)
    except Exception as e:
        print(f"Error in process_email_full: {e}")
        return "low", "general", f"Error: {e}", f"Error: {e}", f"Error: {e}"

if __name__ == "__main__":
    # Ensure database exists
    sql_database()
    
    email_text = """
    Subject: Urgent: Server Down - Immediate Attention Required

    Hi Team,

    I'm writing to report a critical issue with our main production server. It has been down for the last 2 hours, and we're experiencing complete system failure. Our customers are unable to access our services, and we're losing revenue with every minute that passes.

    The error message indicates a database connection failure, but our technical team has been unable to resolve it despite several attempts. We need immediate assistance to get the server back online as soon as possible.

    Thanks,
    John Doe
    """

    print(30 * "+") 
    read_email(email_text)
    
    # Run the full process
    result = process_email_full(email_text)
    
    print(f"\n--- Classification Result ---")
    print(f"Priority: {result[0]}")
    print(f"Category: {result[1]}")
    print(f"Reason: {result[2]}")
    
    print(f"\n--- AI Drafted Reply ---")
    print(result[3])
    
    print(f"\n--- AI Follow-up Schedule ---")
    print(result[4])

    # Print history
    print("\n" + 30 * "+")
    print("SESSION HISTORY:")
    print(history)
    print(30 * "+")
