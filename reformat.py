from get_data import load_customer_data

# Завантаження даних
df = load_customer_data()

df.to_csv("new_data.csv", sep=";")
