import hashlib

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()
