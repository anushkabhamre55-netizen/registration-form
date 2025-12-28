# src/auth.py (patched)
import csv
import bcrypt
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
USERS_PATH = os.path.join(BASE_DIR, "data", "users.csv")

def load_users():
    if not os.path.exists(USERS_PATH):
        return {}
    users = {}
    with open(USERS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # expect columns: email,password_hash
            users[r["email"]] = r["password_hash"]
    return users

def verify_user(email, password):
    users = load_users()
    if email not in users:
        return False
    stored_hash = users[email].encode("utf-8")
    return bcrypt.checkpw(password.encode("utf-8"), stored_hash)
