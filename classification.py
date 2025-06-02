import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from get_data import load_customer_data
import matplotlib.pyplot as plt

def train_churn_model():
    # 1. Завантаження даних
    df = load_customer_data()  # заміни на шлях до твого CSV

    # 2. Попередня обробка
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df.dropna(inplace=True)  # видаляємо рядки з NaN

    # 3. Кодування цільової змінної (Churn: Yes → 1, No → 0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # 4. Кодування категоріальних ознак

    cat_cols = df.select_dtypes(include='object').columns.drop('customerID')
    df = df.drop(columns=['customerID'])  # неінформативна колонка

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 5. Розділення даних на X (ознаки) та y (ціль)
    X = df.drop(columns='Churn')
    y = df['Churn']

    train_x, test_x, train_y , test_y =  train_test_split(X, y, test_size=0.2, random_state=42,stratify=df['Churn'])


    # 7. Побудова моделі
    model = RandomForestClassifier(random_state=42)
    model.fit(train_x, train_y)

    # Прогнози на тесті
    predictions = model.predict(test_x)

    # Вивід звіту класифікації
    print(classification_report(test_y, predictions))
    acc = accuracy_score(test_y, predictions)

    # 7. Переклад назв ознак (тільки для виводу)
    translation = {
        'gender': 'Стать',
        'SeniorCitizen': 'Пенсіонер',
        'Partner': 'Має партнера',
        'Dependents': 'Має утриманців',
        'tenure': 'Тривалість користування',
        'PhoneService': 'Телефонний сервіс',
        'MultipleLines': 'Кілька ліній',
        'InternetService': 'Інтернет сервіс',
        'OnlineSecurity': 'Інтернет захист',
        'OnlineBackup': 'Резервне копіювання',
        'DeviceProtection': 'Захист пристроїв',
        'TechSupport': 'Техпідтримка',
        'StreamingTV': 'СтрімТБ',
        'StreamingMovies': 'СтрімФільми',
        'Contract': 'Тип контракту',
        'PaperlessBilling': 'Безпаперова оплата',
        'PaymentMethod': 'Метод оплати',
        'MonthlyCharges': 'Місячна оплата',
        'TotalCharges': 'Загальна сума'
    }

    # 8. Побудова графіку важливості ознак
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.index = [translation.get(col, col) for col in feature_importances.index]
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title("Важливість ознак у передбаченні відтоку")
    plt.xlabel("Важливість")
    plt.tight_layout()
    return model, plt, label_encoders, acc

train_churn_model()