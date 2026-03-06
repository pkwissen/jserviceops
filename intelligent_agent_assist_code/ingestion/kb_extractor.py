"""
Extract KB number and title from document content
Designed for SharePoint KB format where:

TITLE
KB0015279
Latest Version
Views on the Last 30 Days: 43
Information
"""

import re


def extract_kb_info(text: str, filename: str = "") -> dict:
    """
    Extract KB number and article title from document text.

    Expected format:

    GSD - Outlook: Troubleshooting
    KB0015279
    Latest Version
    Views on the Last 30 Days: 43
    Information

    Returns:
        {
            "kb_number": "KB0015279",
            "kb_title": "KB0015279 – GSD - Outlook: Troubleshooting"
        }
    """

    # --------------------------------------------------
    # Handle explicit "Not Available in Context"
    # --------------------------------------------------
    if re.search(r'Reference:\s*Not Available in Context', text, re.IGNORECASE):
        return {
            "kb_number": "N/A",
            "kb_title": "Not Available in Context"
        }

    # --------------------------------------------------
    # Normalize text
    # --------------------------------------------------
    clean_text = text.replace("\r", "")
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]

    # --------------------------------------------------
    # Step 1: Find KB number anywhere in document
    # --------------------------------------------------
    kb_match = re.search(r'KB\s?\d{4,8}', clean_text, re.IGNORECASE)

    if not kb_match:
        return {
            "kb_number": "Unknown",
            "kb_title": f"{filename or 'Unknown Article'} (kb_number not found)"
        }

    kb_number = kb_match.group(0).replace(" ", "").upper()

    # --------------------------------------------------
    # Step 2: Locate the line containing KB number
    # --------------------------------------------------
    kb_line_index = None

    for idx, line in enumerate(lines):
        if kb_number.lower() in line.lower():
            kb_line_index = idx
            break

    title_line = ""

    # --------------------------------------------------
    # Step 3: Look ABOVE the KB number for actual title
    # --------------------------------------------------
    if kb_line_index is not None and kb_line_index > 0:

        for j in range(kb_line_index - 1, -1, -1):

            candidate = lines[j].strip()

            if not candidate:
                continue

            lower_candidate = candidate.lower()

            # Skip unwanted SharePoint system lines
            if any(skip_word in lower_candidate for skip_word in [
                "latest version",
                "views on",
                "information",
                "last 30 days"
            ]):
                continue

            # Skip lines that contain another KB
            if re.search(r'KB\s?\d{4,8}', candidate, re.IGNORECASE):
                continue

            # Use first clean line above KB
            title_line = candidate
            break

    # --------------------------------------------------
    # Step 4: Fallback if no clean title found
    # --------------------------------------------------
    if not title_line:
        title_line = filename or "Unknown Article"

    # --------------------------------------------------
    # Step 5: Build final formatted title
    # --------------------------------------------------
    kb_title = f"{kb_number} – {title_line}"

    return {
        "kb_number": kb_number,
        "kb_title": kb_title
    }
