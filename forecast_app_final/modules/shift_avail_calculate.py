import pandas as pd

# Load the data (already has Hour, Chat, Phone, Self-service, Sum)
def shift_availability(file_path):
    df = pd.read_excel(file_path)
    # Define shifts (9-hour windows)
    shifts = {
        "06 - 15 IST": list(range(6, 15)),
        "08 - 17 IST": list(range(8, 17)),
        "11 - 20 IST": list(range(11, 20)),
        "12 - 21 IST": list(range(12, 21)),
        "14 - 23 IST": list(range(14, 23)),
        "17 - 02 IST": list(range(17, 24)) + list(range(0, 2)),
        "20 - 05 IST": list(range(20, 24)) + list(range(0, 5)),
        "22 - 07 IST": list(range(22, 24)) + list(range(0, 7)),
    }

    # Compute shift-wise averages
    results = []
    for shift, hrs in shifts.items():
        sel = df[df["Hour"].isin(hrs)]
        results.append({
            "Shift": shift,
            "Tickets": round(sel["Tickets"].mean()) if not sel.empty else 0,
            "Analysts required": round(sel["Analysts required"].mean()) if "Analysts required" in df.columns else 0,
            "Available": round(sel["Analysts available"].mean()) if "Analysts available" in df.columns else 0
        })

    shift_avg_df = pd.DataFrame(results)
    return shift_avg_df