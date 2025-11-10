# 🌞 SMART Solar Optimization & Monitoring System (SOMS)

> ⚡ **An intelligent, data-driven solar plant monitoring and optimization dashboard** — built with **Streamlit, Plotly, XGBoost, and Scikit-learn**.  
> Designed to help solar operators monitor real-time performance, detect faults, and forecast energy generation with AI-driven insights.

---

## 🛰️ Overview

The **SMART Solar Optimization & Monitoring System (SOMS)** is a next-generation platform that integrates **machine learning**, **real-time analytics**, and **solar weather forecasting** to deliver comprehensive insights for solar power plants.

It enables:
- ⚙️ **Operational efficiency monitoring** (AC/DC scaling, inverter performance)
- 🧠 **Fault detection** using intelligent pattern recognition
- 🔮 **Power forecasting** based on weather predictions
- 📊 **Real-time analytics & visualization dashboard**
- ☁️ **Weather-aware insights** for performance optimization

---

## 🧩 Key Features

| Category | Feature | Description |
|-----------|----------|-------------|
| ⚙️ **Monitoring** | Real-time power output visualization | Displays live AC/DC readings with auto-refresh |
| 🧮 **Performance Metrics** | Efficiency, temperature, fault rate, total power | Auto-calculated and calibrated dynamically |
| 🔮 **Forecasting** | Predicts next-day solar output | Based on trained ML model + physics correction |
| ☁️ **Weather Integration** | Uses OpenWeatherMap API *(coming soon)* | For live irradiance, temperature, and wind data |
| 🧠 **Fault Detection** | ML-based inverter fault classification | Detects anomalies from DC/AC imbalance |
| 🧭 **Visual Dashboard** | Interactive Streamlit UI + Plotly graphs | Gradient themes, tabs, metric cards |
| 📈 **Historical Analysis** | Insights from stored data | Identifies trends, peak performance, and efficiency loss |

---

## 🧱 Project Architecture
```bash
📦 SMART-Solar-Optimization-and-Monitoring-System
┣ 📂 dashboard/ # Streamlit dashboard UI
┃ ┗ 📜 soms_dashboard.py
┣ 📂 src/ # Core ML and data modules
┃ ┣ 📜 data_preprocess.py
┃ ┣ 📜 model_train.py
┃ ┣ 📜 forecast_module.py
┃ ┗ 📜 fault_detection.py
┣ 📂 models/ # Saved trained models
┃ ┣ 📜 xgb_with_weather.joblib
┃ ┣ 📜 linear_with_weather.joblib
┃ ┗ 📜 metrics.json
┣ 📂 data/ # Dataset (Generation + Weather)
┃ ┣ 📜 Plant_1_Generation_Data.csv
┃ ┗ 📜 solar_weather.csv
┣ 📂 notebooks/ # Experimental Jupyter notebooks
┣ 📜 train_models.py # Script to retrain models
┣ 📜 LICENSE # MIT License
┗ 📜 README.md # You’re here
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-------------|
| 🖥️ **Frontend/UI** | Streamlit, Plotly, CSS styling |
| 🧠 **AI/ML Models** | XGBoost, Scikit-learn, Statsmodels |
| 🧹 **Data Processing** | Pandas, NumPy |
| 🌤 **Forecasting** | Mock Weather API *(soon → OpenWeatherMap)* |
| 💾 **Persistence** | Joblib for model storage |
| 🧩 **Backend Logic** | Python Modules (`src/`) |
| 📊 **Visualization** | Plotly Express, Plotly Graph Objects |

---

## 🚀 Quick Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/GarvvB/SMART-Solar-Optimization-and-Monitoring-System.git
cd SMART-Solar-Optimization-and-Monitoring-System
```
### 2️⃣ Create a Virtual Environment
```bash
python -m venv soms_env
soms_env\Scripts\activate       # for Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Run the Dashboard
```bash
streamlit run dashboard/soms_dashboard.py
```

🌤 Weather API (Coming Soon)

Integration with OpenWeatherMap for real-time irradiance, temperature, humidity, and wind data.

Live data fusion into forecast and monitoring modules.

Secure API key storage via .env and python-dotenv.

📊 Sample Visuals
Metric Cards	

<img width="1871" height="980" alt="image" src="https://github.com/user-attachments/assets/6140a644-5d0e-45fe-95fc-93c826a6f095" />

Forecast	

<img width="1874" height="839" alt="image" src="https://github.com/user-attachments/assets/c3de1819-c57c-497b-8780-48828ca10294" />

Real-Time Monitor

<img width="1879" height="922" alt="image" src="https://github.com/user-attachments/assets/44c47ace-768c-40c7-8954-c746a0adb7f2" />


🧠 Future Enhancements

☁️ Live weather integration (OpenWeatherMap)

🔋 Multi-inverter comparison dashboard

🕹️ AI-based anomaly prediction (LSTM)

📱 Mobile-friendly responsive Streamlit layout

🌐 Flask/React frontend version for deployment

👨‍💻 Developed By

Garv Bhardwaj
🎓 Developer & AI Enthusiast
🌐 [GitHub Profile](https://github.com/GarvvB)

💡 “Turning renewable data into intelligent decisions.”

📜 License

This project is licensed under the MIT License — see the LICENSE
 file for details.

💬 Feedback & Contributions

Pull requests, suggestions, and issues are welcome!

💡 If you like this project, give it a ⭐ on GitHub.
