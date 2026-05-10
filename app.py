import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Загрузка модели ----------
@st.cache_resource
def load_model():
    model = joblib.load("ridge_model.pkl")
    scaler = joblib.load("ridge_scaler.pkl")
    return model, scaler

model, scaler = load_model()
FEATURE_NAMES = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Предсказание цены автомобиля")

menu = st.sidebar.radio("Меню", ["EDA", "Предсказание", "Важность признаков"])

# ============================================
# 1. EDA (красивые графики)
# ============================================
if menu == "EDA":
    st.header("📊 Разведочный анализ данных")
    try:
        df = pd.read_csv("cars_train.csv")
        # Приводим числовые колонки к float
        for col in FEATURE_NAMES + ['selling_price']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    except:
        st.error("Не найден файл cars_train.csv. Загрузите его в репозиторий.")
        st.stop()

    st.subheader("1. Распределение цены")
    fig, ax = plt.subplots()
    df['selling_price'].dropna().hist(bins=50, alpha=0.7, color='skyblue', edgecolor='black', ax=ax)
    ax.set_title('Selling price distribution')
    st.pyplot(fig)

    st.subheader("2. Зависимость цены от года выпуска")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['year'], df['selling_price'], alpha=0.4, c='green')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Price')
    st.pyplot(fig2)

    st.subheader("3. Цена в зависимости от мощности")
    fig3, ax3 = plt.subplots()
    ax3.scatter(df['max_power'], df['selling_price'], alpha=0.4, c='red')
    ax3.set_xlabel('Max Power (bhp)')
    ax3.set_ylabel('Price')
    st.pyplot(fig3)

    st.subheader("4. Boxplot цены по типу топлива")
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=df, x='fuel', y='selling_price', ax=ax4)
    ax4.set_yscale('log')
    ax4.set_title('Price vs Fuel type (log scale)')
    st.pyplot(fig4)

    st.subheader("5. Корреляционная матрица числовых признаков")
    num_cols = FEATURE_NAMES + ['selling_price']
    corr_data = df[num_cols].dropna()
    if not corr_data.empty:
        corr = corr_data.corr()
        fig5, ax5 = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax5)
        st.pyplot(fig5)
    else:
        st.warning("Недостаточно данных для корреляции.")

# ============================================
# 2. ПРЕДСКАЗАНИЕ (CSV или ручной ввод)
# ============================================
elif menu == "Предсказание":
    st.header("💰 Предсказание цены")
    mode = st.radio("Выберите способ ввода", ["Ручной ввод", "Загрузить CSV файл"])
    
    if mode == "Ручной ввод":
        st.subheader("Введите 8 характеристик автомобиля")
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
            st.success(f"✨ Предсказанная цена: **{pred_price:,.0f} ₽**")
    
    else:  # CSV загрузка
        uploaded = st.file_uploader("Загрузите CSV с колонками: " + ", ".join(FEATURE_NAMES), type="csv")
        if uploaded:
            df_input = pd.read_csv(uploaded)
            # Проверяем наличие всех колонок
            missing = set(FEATURE_NAMES) - set(df_input.columns)
            if missing:
                st.error(f"В файле отсутствуют столбцы: {missing}")
            else:
                X = df_input[FEATURE_NAMES].values
                X_scaled = scaler.transform(X)
                pred_log = model.predict(X_scaled)
                pred_price = np.expm1(pred_log)
                df_input['predicted_price'] = pred_price
                st.dataframe(df_input[['predicted_price']].head())
                st.download_button("Скачать результат с предсказаниями", df_input.to_csv(index=False), "predictions.csv")

# ============================================
# 3. ВИЗУАЛИЗАЦИЯ ВЕСОВ
# ============================================
elif menu == "Важность признаков":
    st.header("📈 Коэффициенты модели (влияние на цену)")
    coef = model.coef_
    df_coef = pd.DataFrame({"Признак": FEATURE_NAMES, "Коэффициент": coef})
    df_coef = df_coef.reindex(df_coef["Коэффициент"].abs().sort_values(ascending=False).index)
    
    fig, ax = plt.subplots(figsize=(10,6))
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_coef["Коэффициент"]]
    bars = ax.barh(df_coef["Признак"], df_coef["Коэффициент"], color=colors)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Коэффициент")
    ax.set_title("Влияние признаков на цену (положительный = рост цены)")
    for bar, val in zip(bars, df_coef["Коэффициент"]):
        ax.text(bar.get_width() + 0.01*max(abs(df_coef["Коэффициент"])),
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    with st.expander("Таблица коэффициентов"):
        st.dataframe(df_coef.style.format({"Коэффициент": "{:.4f}"}))
