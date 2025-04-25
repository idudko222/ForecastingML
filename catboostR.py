from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Предобработка данных
df = pd.read_csv('all_v2.csv')
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

# Выбор фичей
features = ['geo_lat', 'geo_lon', 'region', 'building_type', 'level', 'levels',
            'rooms', 'area', 'kitchen_area', 'object_type', 'year', 'month',
            'level_to_levels', 'area_to_rooms']
X = df[features]
y = df['price']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование целевой переменной
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Указание категориальных признаков
categorical_features = ['region', 'building_type', 'object_type']

# Инициализация и обучение модели CatBoost
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=100,  # Для отображения прогресса
    task_type='CPU',  # Явно указываем использование CPU
    early_stopping_rounds=20  # Ранняя остановка для предотвращения переобучения
)

# Обучение модели
model.fit(
    X_train,
    y_train_log,
    cat_features=categorical_features,
    eval_set=(X_test, y_test_log),
    use_best_model=True
)

# Предсказание и оценка
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")