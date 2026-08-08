import csv
from datetime import datetime

LOG_FILE = "safety_incidents.csv"

def init_log_file():
    """Creates CSV file with column headers"""
    try:
        with open(LOG_FILE, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Incident_Type", "Risk_Level", "Angle"])
    except FileExistsError:
        pass

def log_incident(incident_type, risk_level, angle=0):
    """Saves incident record entry"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, incident_type, risk_level, angle])