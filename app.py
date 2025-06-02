import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analyse import get_type_of_service, get_extented_type_of_service, client_info, advanced_internet, monthly_charges, month_type_payment, total_use_and_pay, contract_info, churn_and_inflow
from classification import train_churn_model

model, feature_importances_fig, label_encoders, acc_score = train_churn_model()

st.title("📊 Аналіз клієнтів телеком-компанії")

# Меню
menu = st.sidebar.radio(
    "🔍 Оберіть розділ аналізу:",
    (
        "📊 Дані про притік та відтік клієнтів",
        "📡 Розподіл типу сервісу",
        "🔍 Деталізація сервісів",
        "👥 Дані про клієнтів",
        "📃 Детальна інформація про контракти",
        "🌐 Використання інтернет-послуг",
        "💵 Розподіл місячної оплати",
        "📈 Оплата за типами контрактів",
        "📅 Розподіл тривалості і суми оплати",
        "🧠 Важливість ознак (ML)"
    )
)

# 1. Відтік клієнтів
if menu == "📊 Дані про притік та відтік клієнтів":
    fig = churn_and_inflow()
    st.pyplot(fig)

# 2. Тип сервісу(Інтернет/Телефон)
elif menu == "📡 Розподіл типу сервісу":
    fig = get_type_of_service()
    st.pyplot(fig)

# 3. Детальніша інфа про сервіси
elif menu == "🔍 Деталізація сервісів":
    fig = get_extented_type_of_service()
    st.pyplot(fig)

# 4. Дані про клієнтів
elif menu == "👥 Дані про клієнтів":
    fig = client_info()
    st.pyplot(fig)

# 5. Детальна інфа про інтернет-сервіси
elif menu == "🌐 Використання інтернет-послуг":
    fig = advanced_internet()
    st.pyplot(fig)

# 6. Місячна оплата
elif menu == "💵 Розподіл місячної оплати":
    fig = monthly_charges()
    st.pyplot(fig)

# 7. Місячна оплата по типу контракту
elif menu == "📈 Оплата за типами контрактів":
    fig = month_type_payment()
    st.pyplot(fig)

# 8. Вся тривалість і оплата
elif menu == "📅 Розподіл тривалості і суми оплати":
    fig = total_use_and_pay()
    st.pyplot(fig)
    
# 9. Інформація про контракти
elif menu == "📃 Детальна інформація про контракти":
    fig = contract_info()
    st.pyplot(fig)

# 10. Графік важливості ознак
elif menu == "🧠 Важливість ознак (ML)":
    st.pyplot(feature_importances_fig)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.sidebar.markdown("---")
predict_mode = st.sidebar.checkbox("🔮 Прогнозувати відтік клієнта")

if predict_mode:
    st.header("🔮 Прогнозування відтоку клієнта")
    st.markdown(f'Точність моделі: {acc_score:.4f}')

    # Форма для вводу користувача
    st.subheader("Введіть дані клієнта:")

    gender = st.radio("Стать", ['Male', 'Female'])
    senior = st.radio("Пенсіонер(1 - так/ 0 - ні)", [0, 1])
    partner = st.radio("Є партнер", ['Yes', 'No'])
    dependents = st.radio("Є утриманці", ['Yes', 'No'])
    tenure = st.slider("Тривалість користування (місяців)", 0, 72, 12)
    phone = st.radio("Телефонний сервіс", ['Yes', 'No'])
    if phone == 'No':
        multi = 'No phone service'
    else:
        multi = st.radio("Декілька ліній", ['Yes', 'No'])
    internet = st.radio("Інтернет-сервіс", ['DSL', 'Fiber optic', 'No'])
    
    if internet == 'No':
        security = 'No internet service'
        backup = 'No internet service'
        device = 'No internet service'
        tech = 'No internet service'
        tv = 'No internet service'
        movies = 'No internet service'
        st.markdown("ℹ️ Інтернет-сервіс відсутній — відповідні послуги буде автоматично встановлено як 'No internet service'.")
    else:
        security = st.radio("Захист в Інтернеті", ['Yes', 'No'])
        backup = st.radio("Онлайн резервне копіювання", ['Yes', 'No'])
        device = st.radio("Захист пристроїв", ['Yes', 'No'])
        tech = st.radio("Технічна підтримка", ['Yes', 'No'])
        tv = st.radio("Стрімінгове ТВ", ['Yes', 'No'])
        movies = st.radio("Стрімінгові фільми", ['Yes', 'No'])

    contract = st.radio("Тип контракту", ['Month-to-month', 'One year', 'Two year'])
    paperless = st.radio("Безпаперова оплата", ['Yes', 'No'])
    payment = st.radio("Тип оплати", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.slider("Місячна оплата", 0.0, 150.0, 70.0)
    total = st.slider("Загальна сума", 0.0, 9000.0, 2000.0)

    # Побудова DataFrame
    input_dict = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multi,
        'InternetService': internet,
        'OnlineSecurity': security,
        'OnlineBackup': backup,
        'DeviceProtection': device,
        'TechSupport': tech,
        'StreamingTV': tv,
        'StreamingMovies': movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    input_df = pd.DataFrame([input_dict])

    # Кодування введених значень
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # Прогноз
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Ймовірність відтоку: {probability:.2%}")
    else:
        st.success(f"✅ Клієнт ймовірно залишиться. Ймовірність відтоку: {probability:.2%}")


