import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

from data_processor import WalmartDataProcessor

TEST_WEEKS = 10

NUMERIC_FEATURES = [
    'IsHoliday', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
    'Year', 'Month', 'WeekOfYear',
    'lag_1', 'lag_2', 'lag_4', 'rolling_mean_4', 'rolling_mean_8',
]


def wmae(y_true, y_pred, is_holiday):
    """Официальная метрика конкурса - праздничные недели весят в 5 раз больше."""
    weights = np.where(is_holiday == 1, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


def time_split(df, test_weeks):
    cutoff = df['Date'].max() - pd.Timedelta(weeks=test_weeks)
    train = df[df['Date'] <= cutoff]
    test = df[df['Date'] > cutoff]
    return train, test


def main():
    processor = WalmartDataProcessor(data_dir='data')
    df = processor.run()

    train_df, test_df = time_split(df, TEST_WEEKS)
    print(f"train: {len(train_df)} строк, test: {len(test_df)} строк")
    print(f"test period: {test_df['Date'].min().date()} - {test_df['Date'].max().date()}")

    # Linear Regression - только числовые фичи + Type через one-hot,
    # без сырых Store/Dept (для линейной модели это были бы бессмысленные номера)
    type_dummies_train = pd.get_dummies(train_df['Type'], prefix='Type')
    type_dummies_test = pd.get_dummies(test_df['Type'], prefix='Type').reindex(
        columns=type_dummies_train.columns, fill_value=0
    )

    X_train_lr = pd.concat([train_df[NUMERIC_FEATURES], type_dummies_train], axis=1)
    X_test_lr = pd.concat([test_df[NUMERIC_FEATURES], type_dummies_test], axis=1)

    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, train_df['Weekly_Sales'])
    lr_pred = lr_model.predict(X_test_lr)

    # CatBoost - добавляем Store и Dept как категориальные признаки напрямую,
    # без one-hot: бустинг сам находит нужные разбиения по каждому магазину/отделу
    cat_features_names = ['Store', 'Dept', 'Type']
    cb_features = NUMERIC_FEATURES + cat_features_names

    X_train_cb = train_df[cb_features].copy()
    X_test_cb = test_df[cb_features].copy()
    for col in cat_features_names:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    cb_model = CatBoostRegressor(
        iterations=500, learning_rate=0.08, depth=8,
        cat_features=cat_features_names, verbose=0, random_seed=42,
    )
    cb_model.fit(X_train_cb, train_df['Weekly_Sales'])
    cb_pred = cb_model.predict(X_test_cb)

    y_test = test_df['Weekly_Sales'].values
    is_holiday_test = test_df['IsHoliday'].values

    print("\n=== МЕТРИКИ ===")
    print(f"Linear Regression - MAE: {mean_absolute_error(y_test, lr_pred):.0f}, "
          f"WMAE: {wmae(y_test, lr_pred, is_holiday_test):.0f}")
    print(f"CatBoost          - MAE: {mean_absolute_error(y_test, cb_pred):.0f}, "
          f"WMAE: {wmae(y_test, cb_pred, is_holiday_test):.0f}")

    importance = cb_model.get_feature_importance()
    print("\n=== FEATURE IMPORTANCE (CatBoost) ===")
    for name, score in sorted(zip(cb_features, importance), key=lambda x: -x[1]):
        print(f"{name}: {score:.2f}")

    # агрегируем по неделям для читаемого графика - на уровне Store-Dept
    # тестовая выборка слишком шумная для визуального сравнения
    plot_df = test_df[['Date']].copy()
    plot_df['actual'] = y_test
    plot_df['catboost'] = cb_pred
    plot_df['linear_regression'] = lr_pred
    weekly = plot_df.groupby('Date').sum()

    plt.figure(figsize=(12, 6))
    plt.plot(weekly.index, weekly['actual'], label='Реальные продажи', color='black', alpha=0.7)
    plt.plot(weekly.index, weekly['catboost'], label='Прогноз CatBoost', color='blue', linestyle='--')
    plt.plot(weekly.index, weekly['linear_regression'], label='Прогноз Linear Regression', color='red', linestyle=':')
    plt.title('Суммарные недельные продажи по всем магазинам - тестовый период')
    plt.ylabel('Weekly Sales, $')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()


if __name__ == '__main__':
    main()
