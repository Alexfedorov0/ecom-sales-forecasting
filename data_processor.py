import numpy as np
import pandas as pd


class WalmartDataProcessor:
    """Загрузка и подготовка данных Walmart Recruiting - Store Sales Forecasting.

    Работает на уровне Store-Dept-Date (как в оригинальном датасете):
    объединяет продажи с внешними регрессорами (погода, цена топлива, CPI,
    безработица, маркдауны) и характеристиками магазинов, затем строит
    лаги и скользящие средние по каждой паре Store-Dept отдельно.
    """

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir

    def load_and_merge(self):
        train = pd.read_csv(f'{self.data_dir}/train.csv', parse_dates=['Date'])
        features = pd.read_csv(f'{self.data_dir}/features.csv', parse_dates=['Date'])
        stores = pd.read_csv(f'{self.data_dir}/stores.csv')

        # IsHoliday дублируется в train и features - оставляем одну версию
        features = features.drop(columns=['IsHoliday'])

        df = train.merge(features, on=['Store', 'Date'], how='left')
        df = df.merge(stores, on='Store', how='left')
        df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
        return df

    def clean(self, df):
        df = df.copy()

        # маркдауны появляются в данных только с ноября 2011 - до этого NaN,
        # это не пропуск в измерении, а отсутствие акции
        markdown_cols = [c for c in df.columns if c.startswith('MarkDown')]
        df[markdown_cols] = df[markdown_cols].fillna(0)

        # CPI и Unemployment пропущены точечно по некоторым магазинам/датам -
        # заполняем ближайшим известным значением по каждому магазину
        for col in ['CPI', 'Unemployment']:
            df[col] = df.groupby('Store')[col].transform(lambda s: s.ffill().bfill())

        # отрицательные продажи (возвраты) погоды не делают, но модели их
        # стоит видеть как есть - только клипуем совсем экстремальные выбросы
        df['Weekly_Sales'] = df['Weekly_Sales'].clip(lower=-5000)

        df['IsHoliday'] = df['IsHoliday'].astype(int)
        return df

    def build_features(self, df):
        df = df.copy()

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

        df['Type'] = df['Type'].astype('category')

        group = df.groupby(['Store', 'Dept'])['Weekly_Sales']
        df['lag_1'] = group.shift(1)
        df['lag_2'] = group.shift(2)
        df['lag_4'] = group.shift(4)

        shifted = df.groupby(['Store', 'Dept'])['lag_1']
        df['rolling_mean_4'] = shifted.transform(lambda s: s.rolling(4).mean())
        df['rolling_mean_8'] = shifted.transform(lambda s: s.rolling(8).mean())

        df = df.dropna(subset=['lag_1', 'lag_2', 'lag_4', 'rolling_mean_4', 'rolling_mean_8'])
        df = df.reset_index(drop=True)
        return df

    def run(self):
        df = self.load_and_merge()
        df = self.clean(df)
        df = self.build_features(df)
        return df
