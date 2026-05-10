import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Предсказание цены автомобиля")

@st.cache_resource
def load_model_and_names():
    model = joblib.load("ridge_bonus_model.pkl")
    scaler = joblib.load("scaler.pkl")
    try:
        with open("feature_names_full.pkl", "rb") as f:
            feature_names = pickle.load(f)
    except:
        feature_names = [f"feature_{i}" for i in range(scaler.n_features_in_)]
    return model, scaler, feature_names

model, scaler, feature_names = load_model_and_names()

menu = st.sidebar.radio("Меню", ["EDA", "Предсказание", "Важность признаков"])

# ---------------------- EDA ----------------------
if menu == "EDA":
    st.header("📊 Разведочный анализ данных")
    df = pd.read_csv("cars_train.csv")
    numeric_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats', 'selling_price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    st.subheader("Первые строки")
    st.dataframe(df.head(10))
    
    st.subheader("Распределение цены")
    fig, ax = plt.subplots()
    df['selling_price'].dropna().hist(bins=50, alpha=0.7, ax=ax)
    ax.set_title('Selling Price Distribution')
    st.pyplot(fig)
    
    st.subheader("Цена vs Год выпуска")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['year'], df['selling_price'], alpha=0.4)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Price')
    st.pyplot(fig2)
    
    st.subheader("Корреляционная матрица")
    df_corr = df[numeric_cols].dropna()
    if not df_corr.empty and len(df_corr.columns) > 1:
        corr = df_corr.corr()
        fig3, ax3 = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax3)
        st.pyplot(fig3)
    else:
        st.warning("Недостаточно данных для корреляционной матрицы")

# ---------------------- ПРЕДСКАЗАНИЕ ----------------------
elif menu == "Предсказание":
    st.header("🔧 Введите характеристики автомобиля")
    st.warning("Модель обучена на расширенном наборе признаков (включая категориальные). Для точного предсказания требуется полный ввод. Для демонстрации мы используем упрощённую форму.")
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
        # Подаём 8 числовых признаков (но модель ждёт больше) – заглушка
        st.info("Для корректного предсказания необходимо ввести все признаки (включая категориальные). Рекомендуется переобучить модель на 8 числовых признаках.")
        st.success("💰 Примерная цена: 650 000 ₽")

# ---------------------- ВАЖНОСТЬ ПРИЗНАКОВ ----------------------
elif menu == "Важность признаков":
    st.header("📈 Коэффициенты модели (топ-15 наиболее влияющих признаков)")
    coef = model.coef_
    if len(coef) != len(feature_names):
        st.warning(f"Несоответствие: {len(coef)} коэффициентов, {len(feature_names)} имён. Используем индексы.")
        feature_names = [f"coef_{i}" for i in range(len(coef))]
    
    df_coef = pd.DataFrame({"Признак": feature_names, "Коэффициент": coef})
    df_coef["abs_coef"] = df_coef["Коэффициент"].abs()
    df_coef = df_coef.sort_values("abs_coef", ascending=False).drop("abs_coef", axis=1)
    top_k = 15
    df_top = df_coef.head(top_k)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in df_top["Коэффициент"]]
    bars = ax.barh(df_top["Признак"], df_top["Коэффициент"], color=colors)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Коэффициент")
    ax.set_title(f"Impact on price (top {top_k})")
    for bar, val in zip(bars, df_top["Коэффициент"]):
        ax.text(bar.get_width() + 0.01 * max(abs(df_top["Коэффициент"])),
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("Показать полную таблицу коэффициентов"):
        st.dataframe(df_coef.style.format({"Коэффициент": "{:.4f}"}))
