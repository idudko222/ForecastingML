import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import os

# 1.1 Оптимизация типов данных и загрузки
dtypes_optimized = {
    'geo_lat': 'float32',
    'geo_lon': 'float32',
    'price': 'float32',
    'rooms': 'int8',
    'building_type': 'int8',
    'object_type': 'int8',
    'level': 'int8',
    'levels': 'int8'
}

# Загрузка данных с оптимизированными типами
start_time = time.time()
df = pd.read_csv('all_v2.csv', dtype=dtypes_optimized)
print(f"Data loading time: {time.time() - start_time:.2f} sec")

# 1.2 Увеличение выборки до 30% с балансировкой (2.3)
df = df.sample(frac=0.3, random_state=42)
df['price_bin'] = pd.qcut(df['price'], q=10, labels=False)

# Предобработка
df = df.drop_duplicates()
df = df.drop('time', axis=1)
df['object_type'] = df['object_type'].apply(lambda x: 2 if x == 11 else x)
df['rooms'] = df['rooms'].apply(lambda x: 0 if x < 0 else x).astype('int8')
df['price'] = df['price'].abs()
df = df[(df['price'] <= 50_000_000) & (df['price'] >= 800_000)]
df.loc[df['level'] > df['levels'], 'level'] = df['levels']
df['level_to_levels'] = df['level'] / df['levels']
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df.drop(['date'], axis=1)
df = df[(df['area'] <= 150) & (df['area'] >= 10)]
df = df[(df['kitchen_area'] <= 40) & (df['kitchen_area'] >= 6)]
df['area_to_rooms'] = (df['area'] / df['rooms']).replace(np.inf, 0)

# 2.1 Фичингениринг
#df['price_per_sqm'] = df['price'] / df['area']
df['room_size'] = df['area'] / df['rooms'].clip(1)
df['is_last_floor'] = (df['level'] == df['levels']).astype('int8')

# Подготовка фичей
features = ['geo_lat', 'geo_lon', 'region', 'building_type', 'level',
            'levels', 'rooms', 'area', 'kitchen_area', 'object_type',
            'year', 'month', 'level_to_levels', 'area_to_rooms',
             'room_size', 'is_last_floor']

# 2.2 Кодирование категориальных признаков
encoder = TargetEncoder(cols=['region'])
X = df[features]
y = df['price']

# Разделение с балансировкой по ценовым корзинам
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=df['price_bin'], random_state=42
)

# Применяем кодировщик только на обучающих данных
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# 1.2 Параллелизация обучения
start_train = time.time()
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,  # Используем все ядра
    max_samples=0.5  # Бэггинг на 50% данных
)
model.fit(X_train_scaled, y_train)
print(f"Training time: {time.time() - start_train:.2f} sec")

# Оценка
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nOptimized Model Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Визуализация
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Optimized Feature Importance')
plt.tight_layout()
plt.show()

# Анализ остатков
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Prediction Error (RUB)')
plt.show()

# Создаём папку, если её нет
os.makedirs("core/ml", exist_ok=True)

# Сохраняем модель, энкодер и скейлер
joblib.dump(model, "core/ml/model.pkl")
joblib.dump(encoder, "core/ml/encoder.pkl")
joblib.dump(scaler, "core/ml/scaler.pkl")

print("✅ Модель, энкодер и скейлер успешно сохранены в папку core/ml/")