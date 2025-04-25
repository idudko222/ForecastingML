import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

dtypes = {
    'geo_lat': 'float32',
    'geo_lon': 'float32',
    'price': 'float32',
    'rooms': 'int8'
}

# Загрузка данных
df = pd.read_csv('all_v2.csv', dtype=dtypes)
df = df.sample(frac=0.1, random_state=42)  # 50% данных

# Предобработка данных (аналогично вашему коду)
df = df.drop_duplicates()
df = df.drop('time', axis=1)
df['object_type'] = df['object_type'].apply(lambda x: 2 if x == 11 else x)
df['rooms'] = df['rooms'].apply(lambda x: 0 if x < 0 else x).astype('int8')
df['price'] = df['price'].abs()
df = df[(df['price'] <= 50000000) & (df['price'] >= 800000)]
df.loc[df['level'] > df['levels'], 'level'] = df['levels']
df['level_to_levels'] = df['level'] / df['levels']
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df.drop(['date'], axis=1)
df = df[(df['area'] <= 150) & (df['area'] >= 10)]
df = df[(df['kitchen_area'] <= 40) & (df['kitchen_area'] >= 6)]
df['area_to_rooms'] = (df['area'] / df['rooms']).replace(np.inf, 0)  # Замена бесконечностей на 0

# Выбор признаков и целевой переменной
features = ['geo_lat', 'geo_lon', 'region', 'building_type', 'level',
            'levels', 'rooms', 'area', 'kitchen_area', 'object_type',
            'year', 'month', 'level_to_levels', 'area_to_rooms']
X = df[features]
y = df['price']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование числовых признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели (RandomForestRegressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test_scaled)

# Оценка точности модели
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Визуализация важности признаков
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Важность признаков')
plt.show()

# Визуализация предсказаний vs реальных значений
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Реальная цена')
plt.ylabel('Предсказанная цена')
plt.title('Предсказания vs Реальные значения')
plt.show()