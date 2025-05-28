import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import os

region_name = {
    '2661': 'Санкт-Петербург',
    '3446': 'Ленинградская область',
    '3': 'Москва',
    '81': 'Московская область',
    '2843': 'Краснодарский край',
    '2871': 'Нижегородская область',
    '3230': 'Ростовская область',
    '3106': 'Самарская область',
    '2922': 'Республика Татарстан',
    '2900': 'Ставропольский край',
    '2722': 'Республика Башкортостан',
    '6171': 'Свердловская область',
    '4417': 'Республика Коми',
    '5282': 'Челябинская область',
    '5368': 'Иркутская область',
    '5520': 'Пермский край',
    '6817': 'Алтайский край',
    '9579': 'Республика Бурятия',
    '2604': 'Ярославская область',
    '1010': 'Удмуртская Республика',
    '7793': 'Псковская область',
    '13919': 'Республика Северная Осетия — Алания',
    '2860': 'Кемеровская область',
    '3019': 'Чувашская Республика',
    '4982': 'Республика Марий Эл',
    '9648': 'Кабардино-Балкарская Республика',
    '5241': 'Республика Мордовия',
    '3870': 'Красноярский край',
    '3991': 'Тюменская область',
    '2359': 'Республика Хакасия',
    '9654': 'Новосибирская область',
    '2072': 'Воронежская область',
    '8090': 'Республика Карелия',
    '4007': 'Республика Дагестан',
    '11171': 'Республика Саха (Якутия)',
    '10160': 'Забайкальский край',
    '7873, 6937': 'Республика Крым',
    '2594': 'Кировская область',
    '8509': 'Республика Калмыкия',
    '11416': 'Республика Адыгея',
    '11991': 'Карачаево-Черкесская Республика',
    '5178': 'Республика Тыва',
    '13913': 'Республика Ингушетия',
    '6309': 'Республика Алтай',
    '5952': 'Белгородская область',
    '6543': 'Архангельская область',
    '2880': 'Тверская область',
    '5993': 'Пензенская область',
    '2484': 'Ханты-Мансийский автономный округ',
    '4240': 'Липецкая область',
    '5789': 'Владимирская область',
    '14880': 'Ямало-Ненецкий автономный округ',
    '1491': 'Рязанская область',
    '2885': 'Чеченская Республика',
    '5794': 'Смоленская область',
    '2528': 'Саратовская область',
    '4374': 'Вологодская область',
    '4695': 'Волгоградская область',
    '2328': 'Калужская область',
    '5143': 'Тульская область',
    '2806': 'Тамбовская область',
    '14368': 'Мурманская область',
    '5736': 'Новгородская область',
    '7121': 'Курская область',
    '4086': 'Хабаровский край',
    '821': 'Брянская область',
    '10582': 'Астраханская область',
    '7896': 'Калининградская область',
    '8640': 'Омская область',
    '5703': 'Курганская область',
    '10201': 'Томская область',
    '4249': 'Ульяновская область',
    '3153': 'Оренбургская область',
    '4189': 'Костромская область',
    '2814': 'Орловская область',
    '13098': 'Камчатский край',
    '8894': 'Ивановская область',
    '7929': 'Амурская область',
    '16705': 'Магаданская область',
    '69': 'Еврейская автономная область',
    '4963': 'Приморский край',
    '1901': 'Сахалинская область',
    '61888': 'Ненецкий автономный округ'
}

# 1.1 Оптимизация типов данных и загрузки
dtypes_optimized = {
    'geo_lat': 'float32',
    'geo_lon': 'float32',
    'price': 'float32',
    'rooms': 'int8',
    'building_type': 'float32',
    'object_type': 'int8',
    'level': 'int8',
    'levels': 'int8'
}


def load_and_preprocess_data(filepath, sample_frac=1.0, random_state=42):
    """Универсальная функция загрузки и предобработки"""
    print(f"\nLoading data from {filepath}...")
    start_time = time.time()

    # Загрузка с возможностью выборки строк
    df = pd.read_csv(filepath, dtype=dtypes_optimized)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    # Предобработка
    df = df.drop_duplicates()
    if 'time' in df.columns:
        df = df.drop('time', axis=1)

    df['object_type'] = df['object_type'].apply(lambda x: 2 if x == 11 else x)
    df['rooms'] = df['rooms'].apply(lambda x: 0 if x < 0 else x).astype('int8')
    df['price'] = df['price'].abs()
    df = df[(df['price'] <= 50_000_000) & (df['price'] >= 800_000)]

    # Обработка даты
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Остальная предобработка
    df.loc[df['level'] > df['levels'], 'level'] = df['levels']
    df['level_to_levels'] = df['level'] / df['levels']
    df = df.drop(['date'], axis=1)
    df = df[(df['area'] <= 150) & (df['area'] >= 10)]
    df = df[(df['kitchen_area'] <= 40) & (df['kitchen_area'] >= 6)]
    df['area_to_rooms'] = (df['area'] / df['rooms']).replace(np.inf, 0)
    df['room_size'] = df['area'] / df['rooms'].clip(1)
    df['is_last_floor'] = (df['level'] == df['levels']).astype('int8')
    df['building_type'] = df['building_type'].fillna(0).astype('int8')

    print(f"Processed {len(df)} samples in {time.time() - start_time:.2f} sec")
    return df


# 1. Загрузка данных
historic_data = load_and_preprocess_data('all_v2.csv', sample_frac=0.05)
current_data = load_and_preprocess_data('test.csv')

# 2. Подготовка фичей
features = ['geo_lat', 'geo_lon', 'region', 'building_type', 'level', 'levels', 'rooms',
            'area', 'kitchen_area', 'object_type', 'year', 'month',
            'level_to_levels', 'area_to_rooms', 'room_size', 'is_last_floor']

# 3. Кодирование и масштабирование
encoder = TargetEncoder(cols=['region'])
scaler = StandardScaler()

X_hist = historic_data[features]
y_hist = historic_data['price']
X_curr = current_data[features]
y_curr = current_data['price']

X_hist_encoded = encoder.fit_transform(X_hist, y_hist)
X_curr_encoded = encoder.transform(X_curr)

X_hist_scaled = scaler.fit_transform(X_hist_encoded)
X_curr_scaled = scaler.transform(X_curr_encoded)

# 4. Обучение базовой модели
print("\nTraining base model...")
base_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_samples=0.5
)
base_model.fit(X_hist_scaled, y_hist)

# 5. Создание и обучение финальной модели
print("\nCreating final model...")

target_transformer = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=0.5))
    ]
)

# 2. Комбинированная модель
final_model = StackingRegressor(
    estimators=[('base', base_model)],
    final_estimator=Ridge(alpha=0.5)
)

final_model.fit(X_curr_scaled, y_curr)

# 6. Оценка модели
X_train, X_test, y_train, y_test = train_test_split(
    X_curr, y_curr, test_size=0.2, random_state=42
)

X_test_scaled = scaler.transform(encoder.transform(X_test))
y_pred = final_model.predict(X_test_scaled)

print("\nModel Metrics:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# 7. Визуализация и сохранение
feature_importance = pd.DataFrame(
    {
        'Feature': features,
        'Importance': base_model.feature_importances_
    }
).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Сохранение моделей
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/model.pkl", compress=3)
joblib.dump(encoder, "models/encoder.pkl", compress=3)
joblib.dump(scaler, "models/scaler.pkl", compress=3)

print("\n✅ Model saved successfully")
