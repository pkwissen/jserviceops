import os

# Detect base directory (folder where app.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data & output directories (auto-create if missing)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
EXCEL_PATH = os.path.join(DATA_DIR, "Questionnaire_Checklist.xlsx")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Questionnaire_submissions.xlsx")

# Sheet name for storing submissions
OUTPUT_SHEET = "Submissions"
