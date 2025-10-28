# Equipment-Failure-Prediction
Predictive Maintenance — Time-to-Failure (RUL) for Oilfield Pump. Predict Remaining Useful Life (RUL) of the pump (in hours)
## 📘 Project Overview
This project demonstrates how **data analytics and machine learning** can be applied to predict the **Remaining Useful Life (RUL)** of an oilfield pump using synthetic sensor data.  
It mimics a real-world **predictive maintenance** use case in the oil & gas sector — enabling proactive maintenance and reduced downtime.

---

## 📂 Files Included
| File | Description |
|------|--------------|
| `equipment_failure_data.csv` | Synthetic dataset of oilfield pump sensor readings |
| `equipment_failure_prediction.ipynb` | Jupyter notebook with full data analysis, model training, and SHAP interpretability |
| `app.py` | Streamlit dashboard app for live RUL prediction and visualization |

---

## ⚙️ Setup Instructions

### 1. Install Dependencies
Ensure Python ≥ 3.8 is installed, then run:
```bash
pip install -r requirements.txt
## If you don’t have a requirements file, install manually:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap plotly streamlit
2. Run the Notebook
Launch the notebook for data exploration and model training:
jupyter notebook equipment_failure_prediction.ipynb

3. Launch the Dashboard
Start the Streamlit app to explore predictions interactively:
streamlit run app.py
Then open the provided URL (usually http://localhost:8501) in your browser.

📊 Key Features
End-to-end workflow: data preprocessing → EDA → model training → SHAP explainability
Predicts Remaining Useful Life (RUL) for each pump instance
Interactive Streamlit dashboard for real-time predictions
Interpretable models with SHAP feature importance plots

🧠 Techniques Used
Regression Modeling (Random Forest, XGBoost)
Cross-validation (KFold)
SHAP Explainability
Interactive Visualization (Plotly, Streamlit)

🛢️ Industry Context

Predictive maintenance is a critical part of Oilfield Digital Transformation.
This workflow can be extended to:

Electric submersible pumps (ESPs)
Compressors
Drilling rig components
Gas lift valves

👨‍💻 Author
Ogar Phil Blaize
Data Analytics Portfolio — Oil & Gas Applications
philemon.rsu@gmail.com
