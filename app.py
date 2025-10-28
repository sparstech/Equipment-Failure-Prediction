
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Pump Predictive Maintenance Dashboard", layout="wide")

st.title("🛠️ Oilfield Pump Predictive Maintenance Dashboard")
st.markdown("This dashboard predicts the **Remaining Useful Life (RUL)** of a pump using sensor data.")

# --- Load dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("equipment_failure_data.csv")

df = load_data()

# --- Sidebar controls ---
st.sidebar.header("Sensor Inputs")
pressure = st.sidebar.slider("Pressure (psi)", 2500, 3500, 3000)
temperature = st.sidebar.slider("Temperature (°F)", 60, 120, 90)
vibration = st.sidebar.slider("Vibration (g)", 0.2, 1.0, 0.5)
flow_rate = st.sidebar.slider("Flow Rate (bbl/day)", 600, 950, 800)
current_draw = st.sidebar.slider("Current Draw (A)", 90, 150, 120)
run_time_hours = st.sidebar.slider("Run Time (hrs)", 100, 2000, 1000)
maintenance_flag = st.sidebar.selectbox("Recent Maintenance?", [0, 1])

# --- Train lightweight model ---
features = ['pressure','temperature','vibration','flow_rate','current_draw','run_time_hours','maintenance_flag']
X = df[features]
y = df['RUL']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Predict RUL for sidebar input ---
input_data = pd.DataFrame([[pressure, temperature, vibration, flow_rate, current_draw, run_time_hours, maintenance_flag]], columns=features)
pred_rul = model.predict(input_data)[0]

st.metric(label="🔮 Predicted Remaining Useful Life (RUL)", value=f"{pred_rul:.1f} hours")

# --- Main visualizations ---
tab1, tab2 = st.tabs(["📈 RUL Distribution", "🧠 Feature Correlation"])

with tab1:
    fig1 = px.histogram(df, x="RUL", nbins=40, title="Distribution of Remaining Useful Life")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    corr = df[features + ['RUL']].corr()
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("Developed for Oil & Gas Predictive Maintenance Portfolio | Powered by Streamlit + scikit-learn")
