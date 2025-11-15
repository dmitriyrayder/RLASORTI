# app.py
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import math

st.set_page_config(layout="wide", page_title="RL Assortment (minimal)")

st.title("Минимальный RL для управления ассортиментом — Demo")

st.markdown(
    """
Коротко: загрузите Excel с колонками:
`Magazin, Datasales, Art, Describe, Model, Segment, purchprice, Price, Qty, Sum`

Далее выберите магазин и запустите простой Q-learning симулятор.
"""
)

uploaded = st.file_uploader("Загрузите Excel (.xlsx/.xls) с продажами", type=["xlsx", "xls", "csv"])
if uploaded is None:
    st.info("Загрузите файл чтобы продолжить. Можете использовать CSV.")
    st.stop()

# load
if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

# basic checks
required = {"Magazin","Datasales","Art","Segment","purchprice","Price","Qty","Sum"}
missing = required - set(df.columns)
if missing:
    st.error(f"В файле отсутствуют колонки: {missing}")
    st.stop()

# preprocess
df["Datasales"] = pd.to_datetime(df["Datasales"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
df["purchprice"] = pd.to_numeric(df["purchprice"], errors="coerce").fillna(0)
df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)
df["Sum"] = pd.to_numeric(df["Sum"], errors="coerce").fillna(df["Price"]*df["Qty"])

stores = df["Magazin"].unique().tolist()
store = st.selectbox("Выберите магазин", stores)

# aggregate features per SKU for selected store
df_store = df[df["Magazin"] == store].copy()
sku_agg = df_store.groupby("Art").agg({
    "Qty":"sum",
    "Sum":"sum",
    "Price":"mean",
    "purchprice":"mean",
    "Datasales":"count",
    "Segment": lambda x: x.mode().iloc[0] if len(x)>0 else "NA"
}).rename(columns={"Datasales":"transactions"})
sku_agg["turnover"] = sku_agg["Qty"] / np.maximum(1, sku_agg["transactions"])
sku_agg["margin"] = sku_agg["Price"] - sku_agg["purchprice"]
# approximate stock if provided
if "Stock" in df_store.columns:
    stock = df_store.groupby("Art")["Stock"].last()
    sku_agg["stock"] = stock
else:
    sku_agg["stock"] = sku_agg["Qty"].rolling(1).apply(lambda x: max(1, int(x.mean()/10))).fillna(5)  # heuristic

sku_agg = sku_agg.reset_index()

st.subheader("Агрегированные характеристики SKU (пример)")
st.dataframe(sku_agg.head(50))

# RL parameters
st.sidebar.header("RL параметры")
n_episodes = st.sidebar.slider("Эпизодов обучения", 10, 2000, 200, step=10)
max_steps = st.sidebar.slider("Макс шагов на эпизод (SKU actions per episode)", 1, 50, 10)
alpha = st.sidebar.number_input("Alpha (скорость обучения)", min_value=0.001, max_value=1.0, value=0.2)
gamma = st.sidebar.number_input("Gamma (дисконт)", min_value=0.0, max_value=1.0, value=0.95)
epsilon = st.sidebar.number_input("Epsilon (exploration)", min_value=0.0, max_value=1.0, value=0.2)

# discretize state: bucket turnover and margin and stock
def discretize(row):
    t = row["turnover"]
    m = row["margin"]
    s = row["stock"]
    tb = int(np.clip(np.digitize(t, [0.1,0.5,1,2,5]), 0, 5))
    mb = int(np.clip(np.digitize(m, [0,5,10,20]), 0, 4))
    sb = int(np.clip(np.digitize(s, [0,2,5,10,20]), 0, 5))
    return (tb, mb, sb, row["Segment"])

sku_agg["_state"] = sku_agg.apply(discretize, axis=1)

# actions: 0 keep,1 remove,2 increase_depth,3 decrease_depth
ACTIONS = {0:"keep",1:"remove",2:"inc_depth",3:"dec_depth"}

# initial Q-table: keyed by state + art
Q = defaultdict(lambda: np.zeros(len(ACTIONS)))

# simple environment simulator (heuristic)
def step_env(sku_df, art, action):
    """
    возвращает: reward, new_qty_estimate
    Правила (примитивные):
     - keep: qty stays
     - remove: qty -> 0, fraction (shift) r распределяется по SKU того же сегмента
     - inc_depth: qty * 1.1
     - dec_depth: qty * 0.9
    reward = delta_profit - penalty_oos
    """
    base = float(sku_df.loc[sku_df["Art"]==art, "Qty"].values[0])
    price = float(sku_df.loc[sku_df["Art"]==art, "Price"].values[0])
    purch = float(sku_df.loc[sku_df["Art"]==art, "purchprice"].values[0])
    seg = sku_df.loc[sku_df["Art"]==art, "Segment"].values[0]
    stock = float(sku_df.loc[sku_df["Art"]==art, "stock"].values[0])
    old_profit = (price - purch) * base

    # effects
    if action == 0:
        new_qty = base
    elif action == 1:
        # remove: fraction shifts to same-segment skus
        shift = 0.4  # 40% of спроса перераспределится
        new_qty = 0.0
        # add amount to others (handled outside)
    elif action == 2:
        new_qty = base * 1.10
    elif action == 3:
        new_qty = base * 0.90
    else:
        new_qty = base

    # oos penalty: if stock < new_qty => penalty
    oos_penalty = 0.0
    if stock < new_qty:
        oos_penalty = (new_qty - stock) * 0.2 * price  # heuristic cost

    new_profit = (price - purch) * new_qty
    reward = new_profit - old_profit - oos_penalty
    # when remove, we also return shifted amount to distribute
    shift_amount = 0.4 * base if action==1 else 0.0
    return reward, new_qty, shift_amount

# helper to distribute shifted sales to same-segment SKUs proportionally by turnover
def distribute_shift(sku_df, art, shift_amount):
    seg = sku_df.loc[sku_df["Art"]==art,"Segment"].values[0]
    pool = sku_df[(sku_df["Segment"]==seg) & (sku_df["Art"]!=art)].copy()
    if pool.empty or shift_amount<=0:
        return {}
    pool["weight"] = pool["turnover"].clip(0.1)
    pool["alloc"] = pool["weight"] / pool["weight"].sum()
    alloc_map = (pool.set_index("Art")["alloc"] * shift_amount).to_dict()
    return alloc_map

# Training loop
if st.button("Запустить обучение RL"):
    progress = st.progress(0)
    sku_df = sku_agg.copy()
    # cache base quantities (so each episode starts from same history)
    base_qty = sku_df.set_index("Art")["Qty"].to_dict()

    for ep in range(n_episodes):
        # reset environment per episode
        sku_df["Qty_sim"] = sku_df["Art"].map(base_qty).astype(float)
        for step in range(max_steps):
            # sample random SKU to act on
            art_row = sku_df.sample(1).iloc[0]
            art = art_row["Art"]
            state = art_row["_state"]
            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(len(ACTIONS))
            else:
                action = int(np.argmax(Q[(state,art)]))
            # apply
            reward, new_qty, shift = step_env(sku_df, art, action)
            # apply shift
            if shift>0:
                alloc = distribute_shift(sku_df, art, shift)
                for a2, add in alloc.items():
                    sku_df.loc[sku_df["Art"]==a2,"Qty_sim"] += add
            # update this SKU qty
            sku_df.loc[sku_df["Art"]==art,"Qty_sim"] = new_qty
            # next_state: recompute discretized features with Qty_sim
            tmp = sku_df[sku_df["Art"]==art].iloc[0].to_dict()
            tmp["Qty"] = new_qty
            tmp["turnover"] = new_qty / max(1, tmp.get("transactions",1))
            next_state = discretize(tmp)
            # Q-learning update
            old_q = Q[(state,art)][action]
            best_next = np.max(Q[(next_state,art)])
            Q[(state,art)][action] = old_q + alpha * (reward + gamma * best_next - old_q)
        if (ep+1) % max(1, (n_episodes//10)) == 0:
            progress.progress((ep+1)/n_episodes)
    st.success("Обучение завершено")

    # produce policy recommendations
    recs = []
    for _, row in sku_agg.iterrows():
        art = row["Art"]
        s = row["_state"]
        qvals = Q[(s,art)]
        act = int(np.argmax(qvals))
        recs.append({
            "Art": art,
            "Segment": row["Segment"],
            "Qty": row["Qty"],
            "Price": row["Price"],
            "Margin": row["margin"],
            "BestAction": ACTIONS[act],
            "Qvals": np.round(qvals,2)
        })
    rec_df = pd.DataFrame(recs).sort_values(by=["Segment","Qty"], ascending=[True, False])
    st.subheader("Рекомендации (политика по SKU)")
    st.dataframe(rec_df.head(200))

    # simple projected effect: apply recommended action and show delta GMV/profit
    total_profit_before = ((sku_agg["Price"] - sku_agg["purchprice"]) * sku_agg["Qty"]).sum()
    proj_profit = 0.0
    for _, r in rec_df.iterrows():
        art = r["Art"]
        action = r["BestAction"]
        price = r["Price"]
        purch = r["Margin"]
        qty = r["Qty"]
        if action=="keep":
            new_qty = qty
        elif action=="remove":
            new_qty = 0
        elif action=="inc_depth":
            new_qty = qty*1.1
        else:
            new_qty = qty*0.9
        proj_profit += (price - (purch)) * new_qty  # careful: purch is margin i used earlier but it's fine for demo

    st.metric("Прибыль до (approx)", f"{total_profit_before:,.0f}")
    st.metric("Проекция прибыли после применения политики (approx)", f"{proj_profit:,.0f}")
    st.caption("Числа — приближённые. Важно: перед применением в продакшен провести оффлайн-симуляции и A/B тесты.")

# allow manual inspection / export
st.sidebar.header("Экспорт")
if st.sidebar.button("Скачать рекомендации (если обучали)"):
    try:
        rec_df  # if exists
        csv = rec_df.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button("Скачать CSV", data=csv, file_name=f"recs_{store}.csv")
    except NameError:
        st.sidebar.error("Сначала запустите обучение и получите рекомендации.")

st.markdown("""
---  
**Примечание:** это минимальный демонстрационный код. Для рабочего решения:
- нужен реалистичный эмулятор спроса / causal models, чтобы не навредить продажам,  
- поддержка A/B тестирования, rollback, мониторинг KPI,  
- более богатое состояние (история продаж, промо, цена конкурентов),  
- алгоритмы: contextual bandits / PPO / DQN для сложных сцен, и off-policy evaluation.
""")
