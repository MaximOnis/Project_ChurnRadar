import pandas as pd

def load_customer_data(filepath: str = 's1.csv') -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, encoding='latin1')  # latin1 уникає decode помилок
        if 'ORDERDATE' in df.columns:
            df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
        return df
    except Exception as e:
        print(f"❌ Помилка при завантаженні даних: {e}")
        return pd.DataFrame()
