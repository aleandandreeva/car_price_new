import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Предсказание цены автомобиля")

@st.cache_resource
def load_model():
    model = joblib.load("ridge_bonus_model.pkl")   # теперь это ridge_8features_model
    scaler = joblib.load("scaler.pkl")             # scaler_8features
    return model, scaler

model, scaler = load_model()

# Имена признаков (точно известны, 8 штук)
feature_names = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']

menu = st.sidebar.radio("Меню", ["EDA", "Предсказание", "Важность признаков"])

# ---------------------- EDA ----------------------
if menu == "EDA":
    st.header("📊 Разведочный анализ данных")
    df = pd.read_csv("cars_train.csv")
    numeric_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats', 'selling_price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    st.dataframe(df.head(10))
    fig, ax = plt.subplots()
    df['selling_price'].dropna().hist(bins=50, alpha=0.7, ax=ax)
    st.pyplot(fig)
    # ... остальные графики (можно оставить как раньше)

# ---------------------- ПРЕДСКАЗАНИЕ ----------------------
elif menu == "Предсказание":
    st.header("🔧 Введите 8 характеристик автомобиля")
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Год выпуска", 1990, 2020, 2015)
        km_driven = st.number_input("Пробег (км)", 0, 500000, 50000)
        mileage = st.number_input("Расход топлива (kmpl)", 0.0, 40.0, 18.0, step=0.5)
        engine = st.number_input("Объём двигателя (CC)", 600, 6000, 1500, step=50)
    with col2:
        max_power = st.number_input("Мощность (bhp)", 30, 800, 100, step=5)
        torque = st.number_input("Крутящий момент (Nm)", 50, 800, 150, step=10)
        max_torque_rpm = st.number_input("Обороты макс. момента (rpm)", 1000, 9000, 2500, step=100)
        seats = st.number_input("Количество мест", 2, 14, 5, step=1)
    
    if st.button("Рассчитать цену"):
        input_arr = np.array([[year, km_driven, mileage, engine, max_power, torque, max_torque_rpm, seats]])
        input_scaled = scaler.transform(input_arr)
        pred_log = model.predict(input_scaled)[0]
        pred_price = np.expm1(pred_log)
        st.success(f"💰 Предсказанная цена: **{pred_price:,.0f} ₽**")

# ---------------------- ВАЖНОСТЬ ПРИЗНАКОВ ----------------------
elif menu == "Важность признаков":
    st.header("📈 Коэффициенты модели (веса признаков)")
    coef = model.coef_
    # Создаём DataFrame с красивыми именами
    df_coef = pd.DataFrame({"Признак": feature_names, "Коэффициент": coef})
    df_coef = df_coef.reindex(df_coef["Коэффициент"].abs().sort_values(ascending=False).index)
    
    fig, ax = plt.subplots(figsize=(10,6))
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_coef["Коэффициент"]]
    bars = ax.barh(df_coef["Признак"], df_coef["Коэффициент"], color=colors)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Коэффициент")
    ax.set_title("Влияние на цену")
    for bar, val in zip(bars, df_coef["Коэффициент"]):
        ax.text(bar.get_width() + 0.01*max(abs(df_coef["Коэффициент"])), 
                bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    with st.expander("Таблица коэффициентов"):
        st.dataframe(df_coef.style.format({"Коэффициент": "{:.4f}"}))
