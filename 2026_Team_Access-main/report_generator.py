import pandas as pd

def generate_summary():
    try:
        df = pd.read_csv("safety_incidents.csv")
        if df.empty:
            return {"Total Incidents": 0, "Posture Alerts": 0, "Hazard Violations": 0}
        
        total = len(df)
        posture_count = len(df[df['Incident_Type'] == 'Poor Ergonomics'])
        hazard_count = len(df[df['Incident_Type'] == 'Hazard Zone Breach'])
        
        return {
            "Total Incidents": total,
            "Posture Alerts": posture_count,
            "Hazard Violations": hazard_count
        }
    except Exception:
        return {"Total Incidents": 0, "Posture Alerts": 0, "Hazard Violations": 0}