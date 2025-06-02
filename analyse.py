# Імпорти
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from get_data import load_customer_data

# Завантаження даних
df = load_customer_data()

#TypeOfService
def get_type_of_service():
    df_copy = df.copy()

    def service_category(row):
        phone = row['PhoneService'] == 'Yes'
        internet = row['InternetService'] != 'No'

        if phone and internet:
            return 'Інтернет + Телефон'
        elif internet:
            return 'Тільки Інтернет'
        elif phone:
            return 'Тільки Телефон'
        else:
            return 'Без послуг'
    
    # Додаємо нову колонку
    df_copy['ServiceCategory'] = df_copy.apply(service_category, axis=1)

    # Підрахунок значень
    counts = df_copy['ServiceCategory'].value_counts()
    labels = counts.index
    sizes = counts.values

    # Побудова кругової діаграми
    colors = ['#66b3ff', '#ffcc99', '#99ff99', '#ff9999']
    plt.figure()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Розподіл клієнтів за типом сервісу')
    plt.axis('equal')  # забезпечує круглу форму
    plt.tight_layout()
    return plt

def get_extented_type_of_service():
    df_copy = df.copy()

    # Тип інтернету
    df_copy['InternetType'] = df_copy['InternetService']

    # Тип телефонного сервісу
    def classify_phone_lines(row):
        if row['PhoneService'] == 'No':
            return 'Немає'
        elif row['MultipleLines'] == 'Yes':
            return 'Кілька ліній'
        else:
            return 'Одна лінія'

    df_copy['PhoneLineType'] = df_copy.apply(classify_phone_lines, axis=1)

    # Фільтрація значень без сервісів
    df_internet = df_copy[df_copy['InternetType'] != 'No']
    df_phone = df_copy[df_copy['PhoneLineType'] != 'Немає']

    # Побудова трьох кругових діаграм
    fig, axes = plt.subplots(1, 2, figure=(12, 5))

    # Діаграма 1: Тип інтернету
    df_internet['InternetType'].value_counts().plot.pie(
        ax=axes[0], autopct='%1.1f%%', startangle=90, colors=['#c2c2f0', '#ffb3e6', '#b3ffcc']
    )
    axes[0].set_title('Тип інтернету')
    axes[0].axis('equal')
    axes[0].set_ylabel('')

    # Діаграма 1: Телефонні лінії
    df_phone['PhoneLineType'].value_counts().plot.pie(
        ax=axes[1], autopct='%1.1f%%', startangle=90, colors=['#f0e68c', '#d3d3d3', '#ffa07a']
    )
    axes[1].set_title('Тип телефонного сервісу')
    axes[1].axis('equal')
    axes[1].set_ylabel('')

    plt.tight_layout()
    return plt

def churn_and_inflow():
    df_copy = df.copy()

    # Класифікація: хто залишився, хто пішов
    def classify_churn(row):
        if row['Churn'] == "No":
            return 'Залишився'
        elif row['Churn'] == "Yes":
            return 'Пішов'
        else:
            return 'Невідомо'

    df_copy['ChurnLabel'] = df_copy.apply(classify_churn, axis=1)

    # Створення ознаки "Новий клієнт"
    df_copy['NewCustomer'] = df_copy['tenure'].apply(lambda x: 'Новий' if x == 1 else 'Старий')

    # Функція для підпису з кількістю та відсотком
    def make_autopct(values):
        def autopct(pct):
            total = sum(values)
            count = int(round(pct * total / 100.0))
            return f'{pct:.1f}%\n({count})'
        return autopct

    # Дві кругові діаграми
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Відтік клієнтів
    churn_counts = df_copy['ChurnLabel'].value_counts()
    colors_churn = ["#60a060", "#ff9999"]
    axes[0].pie(churn_counts, 
                labels=churn_counts.index, 
                autopct=make_autopct(churn_counts), 
                startangle=90, 
                colors=colors_churn)
    axes[0].set_title("Відтік клієнтів")
    axes[0].axis('equal')

    # Притік нових клієнтів
    inflow_counts = df_copy['NewCustomer'].value_counts()
    colors_inflow = ["#66b3ff", "#ffcc99"]
    axes[1].pie(inflow_counts, 
                labels=inflow_counts.index, 
                autopct=make_autopct(inflow_counts), 
                startangle=90, 
                colors=colors_inflow)
    axes[1].set_title("Нові клієнти")
    axes[1].axis('equal')

    plt.tight_layout()
    return plt

def client_info():
    df_copy = df.copy()
    # Словник для назв та колонок
    attributes = {
        'gender': 'Стать клієнта',
        'Churn': 'Відтік клієнтів',
        'SeniorCitizen': 'Чи є пенсіонером'
    }

    # Тип контракту
    def classify_gender(row):
        if row['gender'] == 'Female':
            return 'Жінка'
        elif row['gender'] == 'Male':
            return 'Чоловік'
        else:
            return 'Невідомо'
        
    # Тип оплати
    def classify_age(row):
        if row['SeniorCitizen'] == 0:
            return 'Не пенсіонер'
        elif row['SeniorCitizen'] == 1:
            return 'Пенсіонер'
        else:
            return 'Невідомо'

    df_copy['Gender'] = df_copy.apply(classify_gender, axis=1)
    df_copy['Senior'] = df_copy.apply(classify_age, axis=1)

    # Побудова трьох кругових діаграм
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Діаграма 1: Тип інтернету
    df_copy['Gender'].value_counts().plot.pie(
        ax=axes[0], autopct='%1.1f%%', startangle=90
    )
    axes[0].set_title('Стать')
    axes[0].axis('equal')
    axes[0].set_ylabel('')

    # Діаграма 1: Телефонні лінії
    df_copy['Senior'].value_counts().plot.pie(
        ax=axes[1], autopct='%1.1f%%', startangle=90
    )
    axes[1].set_title('Доля пенсіонерів')
    axes[1].axis('equal')
    axes[1].set_ylabel('')

    plt.tight_layout()
    return plt


def contract_info():
    df_copy = df.copy()

    # Тип контракту
    def classify_contract_type(row):
        if row['Contract'] == 'One year':
            return 'Річний'
        elif row['Contract'] == 'Month-to-month':
            return 'Щомісячний'
        elif row['Contract'] == 'Two year':
            return 'Двохрічний'
        else:
            return 'Невідомо'
        
    # Тип оплати
    def classify_payment_type(row):
        if row['PaymentMethod'] == 'Credit card (automatic)':
            return 'Кредитна картка(авто)'
        elif row['PaymentMethod'] == 'Electronic check':
            return 'Електронний чек'
        elif row['PaymentMethod'] == 'Mailed check':
            return 'Паперовий чек'
        elif row['PaymentMethod'] == 'Bank transfer (automatic)':
            return 'Банківський переказ(авто)'
        else:
            return 'Невідомо'

    df_copy['ContractType'] = df_copy.apply(classify_contract_type, axis=1)
    df_copy['PaymentType'] = df_copy.apply(classify_payment_type, axis=1)

    # Побудова трьох кругових діаграм
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Діаграма 1: Тип інтернету
    df_copy['ContractType'].value_counts().plot.pie(
        ax=axes[0], autopct='%1.1f%%', startangle=90
    )
    axes[0].set_title('Тип контракту')
    axes[0].axis('equal')
    axes[0].set_ylabel('')

    # Діаграма 1: Телефонні лінії
    df_copy['PaymentType'].value_counts().plot.pie(
        ax=axes[1], autopct='%1.1f%%', startangle=90
    )
    axes[1].set_title('Тип оплати')
    axes[1].axis('equal')
    axes[1].set_ylabel('')

    plt.tight_layout()
    return plt

def advanced_internet():
    df_copy = df.copy()

    # Список сервісів
    services = {
        'OnlineSecurity': 'Інтернет-захист',
        'OnlineBackup': 'Резервне копіювання',
        'DeviceProtection': 'Захист пристроїв',
        'TechSupport': 'Тех. підтримка',
        'StreamingTV': 'Стрімінгове ТБ',
        'StreamingMovies': 'Стрімінгові фільми'
    }

    # Підрахунок користування
    usage_data = {}
    for service, name in services.items():
        counts = df_copy[service].value_counts()
        usage_data[name] = {
            'Так': counts.get('Yes', 0),
            'Ні': counts.get('No', 0)
        }

    # Побудова DataFrame для візуалізації
    usage_df = pd.DataFrame(usage_data).T
    usage_df.plot(kind='bar', stacked=True, figsize=(8, 6), color=['#66c2a5', '#fc8d62'])

    plt.title('Розподіл користування послугами')
    plt.ylabel('Кількість клієнтів')
    plt.xlabel('Послуги')
    plt.legend(title='Користується')
    plt.tight_layout()
    return plt

def monthly_charges():
    df_copy = df.copy()

    # Побудова гістограми
    plt.figure()
    plt.hist(df_copy['MonthlyCharges'], bins=30, color='#66b3ff', edgecolor='black')
    plt.title('Розподіл місячної оплати')
    plt.xlabel('Місячна оплата ($)')
    plt.ylabel('Кількість клієнтів')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt

def month_type_payment():
    df_copy = df.copy()

    payment_methods = df_copy['Contract'].unique()

    plt.figure(figsize=(12, 10))

    for i, method in enumerate(payment_methods):
        plt.subplot(2, 2, i + 1)
        subset = df_copy[df_copy['Contract'] == method]
        plt.hist(subset['MonthlyCharges'], bins=30, color='#99ccff', edgecolor='black')
        plt.title(f'Розподіл місячної оплати:\n{method}', fontsize=10)
        plt.xlabel('Місячна оплата ($)+\n')
        plt.ylabel('К-сть клієнтів')
        plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    return plt   

def total_use_and_pay():
    df_copy = df.copy()
    df_copy['TotalCharges'] = pd.to_numeric(df_copy['TotalCharges'], errors='coerce')

    plt.figure(figsize=(13, 8))

    # 1. Тривалість користування (tenure)
    plt.subplot(1, 2, 1)
    plt.hist(df_copy['tenure'], bins=30, color='#66c2a5', edgecolor='black')
    plt.title('Розподіл тривалості надання послуг (tenure)')
    plt.xlabel('Місяці користування')
    plt.ylabel('Кількість клієнтів')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 2. Загальна сума виплат (TotalCharges)
    plt.subplot(1, 2, 2)
    plt.hist(df_copy['TotalCharges'].dropna(), bins=30, color='#fc8d62', edgecolor='black')
    plt.title('Розподіл загальної суми виплат (TotalCharges)')
    plt.xlabel('Сума ($)')
    plt.ylabel('Кількість клієнтів')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    return plt
