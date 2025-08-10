import sqlite3
import os
def safe_json_loads(s):
    try:
        return json.loads(s)
    except Exception:
        return s  # fallback: return raw string if JSON parsing fails

DB_PATH = r"C:\Users\dell\Documents\BITS_STUDY_MATERIAL\Semester - 3\Mlops\assignment_california_housing\california-housing-mlflow\prediction_logs.db"

print(DB_PATH)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT * FROM prediction_logs")
rows = cursor.fetchall()
logs = []
for row in rows:
    logs.append({
                    "timestamp": row[1],
                    "request_data": safe_json_loads(row[2]) if row[2] else None,
                    "prediction": safe_json_loads(row[3]) if row[3] else None,
                    "status_code": row[4],
                    "process_time_ms": row[5]
                })
    print(logs)
conn.close()