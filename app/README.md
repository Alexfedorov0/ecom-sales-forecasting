---
title: Walmart Sales Forecasting
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

# Прогноз недельных продаж - Walmart

Демо модели, которая прогнозирует недельные продажи по паре магазин-отдел
на датасете Walmart Recruiting - Store Sales Forecasting (Kaggle).

Выбираешь магазин, отдел и неделю - приложение показывает реальные продажи,
прогноз CatBoost и историю по этой паре магазин-отдел.

## Модель

CatBoost против Linear Regression, метрика - WMAE (праздничные недели весят
в 5 раз больше обычных). Признаки - лаги продаж, скользящие средние,
погода, цена топлива, CPI, безработица, маркдауны, флаг праздника.
Store и Dept идут в CatBoost как категориальные признаки напрямую, без one-hot.

| Модель            | MAE  | WMAE |
|-------------------|------|------|
| Linear Regression | 1720 | 1851 |
| CatBoost          | 1395 | 1570 |

Код обучения и обработки данных - в основном репозитории проекта
[ecom-sales-forecasting](https://github.com/), датасет с
[Kaggle](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting).

## Локальный запуск

```
pip install -r requirements.txt
streamlit run app.py
```

Модель и обработанные данные уже лежат в `model/` и `data/`, обучать заново не нужно.
