# 2026_Team_Access
# 🛡️ ErgoGuard AI
> **Worker Safety & Ergonomic Risk Monitoring System**

ErgoGuard AI is a real-time computer vision application designed to monitor workplace ergonomics and enforce hazardous area geofencing. Built with Python, OpenCV, and Streamlit, it dynamically computes worker posture angles and tracks safety breaches in real time to prevent injuries and maintain compliance records.

---

## ✨ Key Features

* **Real-time Posture Tracking:** Computes vector angles (Shoulder-Hip-Knee) to assess ergonomic risk dynamically.
* **Hazard Geofencing:** Monitors active danger zones and detects perimeter breaches in real time.
* **Live Monitoring HUD:** Streamlit-powered dashboard displaying live camera feed, KPI metrics, and system status.
* **Automated Incident Logging:** Logs safety breaches and poor ergonomics incidents with rate-limited cooldown timers.
* **Aggregated Compliance Reports:** Real-time JSON and statistical summary generation for safety audits.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **Frontend / Dashboard:** Streamlit
* **Data Processing & Analytics:** NumPy, Pandas

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure Python is installed on your system.

### 2. Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
pip install -r requirements.txt
