# Teen Phone Addiction Prediction Dashboard

A full-stack machine learning project that predicts teen phone addiction levels based on lifestyle and behavioral survey data. It includes:

- 📊 Interactive **Streamlit dashboard** for data exploration
- 🤖 **FastAPI server** for real-time prediction
- 🧪 **MLFlow experiment tracking**
- 🐳 Docker-ready and deployable to cloud

---

## 🚀 Live Demo

🔗 **Coming soon at [mattserdukoff.com](https://mattserdukoff.com)**

---

## 💡 Features

### 🧠 Machine Learning
- Model: `RandomForestRegressor`
- Target: `Addiction_Level` (float, range 0–10)
- Metrics: MSE, RMSE, R², MAE, MAPE, Max Error, Median AE

### 📈 Streamlit Dashboard
- Data filtering and EDA
- Prediction playground
- Visualization of feature importance + prediction accuracy

### ⚙️ FastAPI Backend
- `/predict` endpoint
- Loads latest model from MLFlow automatically
- Works locally and in Docker

---

## 📁 Project Structure

```bash
.
├── data/                        # Raw CSV data
├── app/                         # FastAPI server
│   ├── main.py
│   ├── model.py
│   └── schemas.py
├── train/                       # Model training script
│   └── train_model.py
├── streamlit_dashboard.py       # Dashboard UI
├── requirements.txt
├── README.md
└── mlruns/                      # MLflow local store
```

---

## 📦 Installation

```bash
# Create virtual env
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Usage

### 🧠 Train Model
```bash
python train/train_model.py
```

### ⚙️ Run API
```bash
uvicorn app.main:app --reload
```

### 📊 Run Dashboard
```bash
streamlit run streamlit_dashboard.py
```

---

## 🛠 Requirements

See `requirements.txt` or install manually:
```bash
pip install pandas scikit-learn fastapi uvicorn mlflow streamlit pydantic seaborn matplotlib requests
```

---

## 📸 Screenshots

![dashboard preview](https://via.placeholder.com/800x400?text=Streamlit+Dashboard)

---

## 📌 Credits

- Built by [Matt Serdukoff](https://mattserdukoff.com)
- Dataset: Kaggle – Teen Phone Addiction Survey

---

## 📬 Contact
Have questions? Email me at `m.serdukoff@gmail.com` or [contact me](https://mattserdukoff.com/#contact).
