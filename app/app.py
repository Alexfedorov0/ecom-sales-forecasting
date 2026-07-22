import json

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

st.set_page_config(page_title="Прогноз продаж Walmart", page_icon="📈", layout="wide")

NUMERIC_FEATURES = [
    "IsHoliday", "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "Year", "Month", "WeekOfYear",
    "lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_mean_8",
]
CAT_FEATURES = ["Store", "Dept", "Type"]
FEATURE_COLS = NUMERIC_FEATURES + CAT_FEATURES


@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("model/catboost_model.cbm")
    return model


@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed.parquet")
    return df


@st.cache_data
def load_metrics():
    with open("model/metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_importance():
    return pd.read_csv("model/feature_importance.csv")


def predict_row(model, row):
    x = row[FEATURE_COLS].copy()
    for c in CAT_FEATURES:
        x[c] = str(x[c])
    x = pd.DataFrame([x])
    return float(model.predict(x)[0])


model = load_model()
df = load_data()
metrics = load_metrics()
importance = load_importance()

st.title("Прогноз недельных продаж - Walmart")
st.caption(
    "Датасет Walmart Recruiting - Store Sales Forecasting (Kaggle), "
    f"{metrics['n_stores']} магазинов, {metrics['n_depts']} отделов, "
    f"{metrics['n_rows']:,} наблюдений за 2010-2012".replace(",", " ")
)

with st.sidebar:
    st.header("Параметры")
    store_list = sorted(df["Store"].unique())
    store = st.selectbox("Магазин", store_list, index=0)

    dept_list = sorted(df.loc[df["Store"] == store, "Dept"].unique())
    dept = st.selectbox("Отдел", dept_list, index=0)

    subset = df[(df["Store"] == store) & (df["Dept"] == dept)].sort_values("Date")
    date_options = subset["Date"].dt.date.tolist()
    picked_date = st.selectbox("Неделя", date_options, index=len(date_options) - 1)

    st.markdown("---")
    st.caption("Модель обучена на данных до " + metrics["test_start"] + ", ниже показана оценка на отложенных неделях")

row = subset[subset["Date"].dt.date == picked_date].iloc[0]
pred = predict_row(model, row)
actual = row["Weekly_Sales"]
error = pred - actual

col1, col2, col3, col4 = st.columns(4)
col1.metric("Реальные продажи", f"${actual:,.0f}")
col2.metric("Прогноз модели", f"${pred:,.0f}")
col3.metric("Ошибка", f"${error:,.0f}", delta=f"{error / actual * 100:.1f}%" if actual else None)
col4.metric("Праздничная неделя", "да" if row["IsHoliday"] else "нет")

st.subheader(f"История продаж - магазин {store}, отдел {dept}")

chart_df = subset[["Date", "Weekly_Sales"]].rename(columns={"Weekly_Sales": "Продажи"}).set_index("Date")
st.line_chart(chart_df)

st.caption(
    "Точка на графике не выделена отдельно - смотри значения выше для выбранной недели. "
    "Прогноз строится по лагам и скользящим средним этой же пары магазин-отдел, "
    "поэтому для первых недель истории (нет предыдущих 8 недель) данных на графике нет."
)

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.subheader("Качество модели на тестовом периоде")
    st.caption(f"{metrics['test_start']} - {metrics['test_end']}, {metrics['test_weeks']} недель")
    comparison = pd.DataFrame(
        {
            "Модель": ["Linear Regression", "CatBoost"],
            "MAE": [metrics["linear_regression_mae"], metrics["catboost_mae"]],
            "WMAE": [metrics["linear_regression_wmae"], metrics["catboost_wmae"]],
        }
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption(
        "WMAE - официальная метрика конкурса, праздничные недели весят в 5 раз больше обычных"
    )

with right:
    st.subheader("Feature importance (CatBoost)")
    top_importance = importance.head(10).set_index("feature")
    st.bar_chart(top_importance)

with st.expander("Как устроен прогноз"):
    st.markdown(
        """
CatBoost получает Store и Dept как категориальные признаки напрямую, без one-hot кодирования,
и находит специфику конкретных магазинов и отделов, которую линейная регрессия не ловит.

Признаки для каждой недели - лаги продаж (1, 2, 4 недели назад), скользящие средние
за 4 и 8 недель, погода, цена топлива, CPI, безработица, маркдауны (акции), флаг праздничной недели.
Лаги и скользящие считаются отдельно для каждой пары магазин-отдел, чтобы не подмешивать историю чужого отдела.

Код обучения и обработки данных - в репозитории [ecom-sales-forecasting](https://github.com).
        """
    )
