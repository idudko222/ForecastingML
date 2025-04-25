import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

dtypes = {
    'geo_lat': 'float32',
    'geo_lon': 'float32',
    'price': 'float32',
    'rooms': 'int8'
}

# Загрузка и предобработка данных
df = pd.read_csv('all_v2.csv', dtype=dtypes)
df = df.sample(frac=0.2, random_state=42)  # Берем 50% данных
df = df.drop_duplicates()
df = df.drop('time', axis=1)
df['object_type'] = df['object_type'].apply(lambda x: 2 if x == 11 else x)
df['rooms'] = df['rooms'].apply(lambda x: 0 if x < 0 else x)
df = df[(df['price'] >= 1_000_000) & (df['price'] <= 50_000_000)]
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
df['area_to_rooms'] = (df['area'] / df['rooms']).replace(np.inf, 0)

# Выбор признаков
features = ['geo_lat', 'geo_lon', 'region', 'building_type', 'level', 'levels',
            'rooms', 'area', 'kitchen_area', 'object_type', 'year', 'month',
            'level_to_levels', 'area_to_rooms']
X = df[features]
y = df['price']

# Разделение данных (80% train, 20% test от выбранных 50%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование целевой переменной в логарифмическую шкалу
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Указание категориальных признаков
categoricals = ['region', 'building_type', 'object_type']
for col in categoricals:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Обучение модели
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=50,
    random_state=42
)
model.fit(X_train, y_train_log,
          eval_set=[(X_test, y_test_log)],
          eval_metric='mae',
          categorical_feature=categoricals)

# Предсказание и обратное преобразование из логарифмической шкалы
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

# Оценка качества
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")