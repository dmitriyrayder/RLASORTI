"""
Профессиональная система оптимизации ассортимента SKU на основе Reinforcement Learning

Автор: Data Science Team
Версия: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from io import StringIO
import sys

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import SalesDataLoader
from src.features import SKUFeatureEngineering, SegmentAnalyzer
from src.environment import SKUEnvironment
from src.agents import DQNAgent
from src.metrics import PerformanceTracker
from src.visualization import AssortmentVisualizer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Конфигурация Streamlit
st.set_page_config(
    layout="wide",
    page_title="SKU Optimization System - RL",
    page_icon="📊"
)

# Стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<div class="main-header">🎯 SKU Optimization System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Reinforcement Learning для умного управления ассортиментом</div>',
    unsafe_allow_html=True
)

# Sidebar - Конфигурация
st.sidebar.header("⚙️ Конфигурация")

# Инициализация session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None


# ============= 1. ЗАГРУЗКА ДАННЫХ =============
st.header("📂 1. Загрузка данных")

uploaded_file = st.file_uploader(
    "Загрузите Excel или CSV файл с продажами",
    type=["xlsx", "xls", "csv"],
    help="Файл должен содержать колонки: Magazin, Datasales, Art, Segment, purchase_price, Price, Qty, Sum"
)

if uploaded_file is not None:
    try:
        with st.spinner("Загрузка и обработка данных..."):
            # Сохранить временно файл
            temp_path = Path(f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Загрузка данных
            data_loader = SalesDataLoader()
            df = data_loader.load(str(temp_path))

            # Удалить временный файл
            temp_path.unlink()

            st.session_state.data_loader = data_loader
            st.session_state.df = df
            st.session_state.data_loaded = True

        st.success(f"✅ Загружено {len(df)} записей, {df['Art'].nunique()} уникальных SKU")

        # Показать сводку
        col1, col2, col3, col4 = st.columns(4)
        summary = data_loader.get_summary_stats()

        with col1:
            st.metric("Общий GMV", f"{summary['total_gmv']:,.0f} ₽")
        with col2:
            st.metric("Магазинов", summary['unique_stores'])
        with col3:
            st.metric("Сегментов", summary['unique_segments'])
        with col4:
            st.metric("SKU", summary['unique_skus'])

        # Превью данных
        with st.expander("📊 Превью данных"):
            st.dataframe(df.head(100), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Ошибка при загрузке данных: {str(e)}")
        logger.error(f"Ошибка загрузки: {e}", exc_info=True)
        st.session_state.data_loaded = False

else:
    st.info("👆 Загрузите файл с данными для начала работы")
    st.stop()


# ============= 2. ВЫБОР МАГАЗИНА И НАСТРОЙКИ =============
if st.session_state.data_loaded:
    st.header("🏪 2. Выбор магазина и параметры")

    stores = st.session_state.data_loader.get_summary_stats()['stores']
    selected_store = st.selectbox("Выберите магазин для оптимизации:", stores)

    # Параметры RL
    st.sidebar.subheader("🤖 Параметры RL агента")
    n_episodes = st.sidebar.slider("Количество эпизодов обучения", 10, 1000, 200, step=10)
    max_steps_per_episode = st.sidebar.slider("Макс. шагов на эпизод", 10, 100, 50, step=5)
    learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    gamma = st.sidebar.slider("Gamma (дисконт)", 0.90, 0.99, 0.95, step=0.01)
    epsilon_start = st.sidebar.slider("Epsilon (начальный)", 0.5, 1.0, 1.0, step=0.05)
    epsilon_min = st.sidebar.slider("Epsilon (минимальный)", 0.01, 0.2, 0.05, step=0.01)

    # Получить данные по магазину
    store_df = st.session_state.data_loader.get_store_data(selected_store)
    sku_agg = st.session_state.data_loader.get_sku_aggregates(selected_store)

    st.info(f"📦 В магазине **{selected_store}** доступно **{len(sku_agg)} SKU** для оптимизации")

    # Показать топ SKU
    with st.expander("🔝 Топ-20 SKU по GMV"):
        top_skus = sku_agg.nlargest(20, 'Sum_sum')[
            ['Art', 'Segment_<lambda>', 'Sum_sum', 'Qty_sum', 'margin_mean', 'num_transactions']
        ].rename(columns={
            'Segment_<lambda>': 'Segment',
            'Sum_sum': 'GMV',
            'Qty_sum': 'Quantity',
            'margin_mean': 'Avg Margin'
        })
        st.dataframe(top_skus, use_container_width=True)


# ============= 3. ОБУЧЕНИЕ МОДЕЛИ =============
if st.session_state.data_loaded:
    st.header("🧠 3. Обучение RL модели")

    if st.button("🚀 Запустить обучение DQN агента", type="primary"):
        try:
            with st.spinner("Обучение модели... Это может занять несколько минут."):
                # Feature engineering
                feature_eng = SKUFeatureEngineering(scaler_type='robust')
                sku_with_features = feature_eng.engineer_features(sku_agg, fit=True)

                # Создание environment
                env = SKUEnvironment(
                    sku_df=sku_with_features,
                    feature_engineer=feature_eng,
                    max_steps=max_steps_per_episode
                )

                # Создание агента
                agent = DQNAgent(
                    state_dim=env.state_dim,
                    action_dim=env.action_dim,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    epsilon=epsilon_start,
                    epsilon_min=epsilon_min,
                    epsilon_decay=0.995,
                    buffer_size=5000,
                    batch_size=64
                )

                # Трекер прогресса
                tracker = PerformanceTracker()

                # Прогресс бар
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_container = st.empty()

                # Обучение
                for episode in range(n_episodes):
                    state = env.reset()
                    episode_reward = 0
                    episode_losses = []

                    for step in range(max_steps_per_episode):
                        # Выбор действия
                        action = agent.select_action(state, training=True)

                        # Шаг в среде
                        next_state, reward, done, info = env.step(action)
                        episode_reward += reward

                        # Обучение агента
                        train_metrics = agent.train_step(state, action, reward, next_state, done)
                        if train_metrics['loss'] > 0:
                            episode_losses.append(train_metrics['loss'])

                        state = next_state

                        if done:
                            break

                    # Обновление epsilon
                    agent.update_epsilon()

                    # Записать результаты
                    final_metrics = env.current_metrics
                    tracker.record_episode(episode_reward, final_metrics)

                    # Обновить прогресс
                    progress = (episode + 1) / n_episodes
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Эпизод {episode + 1}/{n_episodes} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Epsilon: {agent.epsilon:.3f} | "
                        f"Avg Loss: {np.mean(episode_losses) if episode_losses else 0:.4f}"
                    )

                    # Показать промежуточные результаты каждые 20 эпизодов
                    if (episode + 1) % 20 == 0:
                        summary = tracker.get_summary()
                        with metrics_container.container():
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Avg Reward", f"{summary['avg_reward']:.2f}")
                            col2.metric("Best Reward", f"{summary['best_reward']:.2f}")
                            col3.metric("Last 10 Avg", f"{summary['last_10_avg_reward']:.2f}")

                # Сохранить результаты
                st.session_state.agent = agent
                st.session_state.env = env
                st.session_state.tracker = tracker
                st.session_state.feature_eng = feature_eng
                st.session_state.sku_with_features = sku_with_features
                st.session_state.model_trained = True

            st.success("✅ Обучение завершено!")

            # Итоговая сводка
            summary = tracker.get_summary()
            st.subheader("📊 Результаты обучения")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Всего эпизодов", summary['total_episodes'])
            col2.metric("Средний Reward", f"{summary['avg_reward']:.2f}")
            col3.metric("Лучший Reward", f"{summary['best_reward']:.2f}")
            col4.metric("Последние 10 эп.", f"{summary['last_10_avg_reward']:.2f}")

            # График обучения
            episodes, rewards = tracker.get_learning_curve()
            visualizer = AssortmentVisualizer(use_plotly=True)
            fig = visualizer.plot_learning_curve(episodes, rewards, title="Learning Curve - DQN Agent")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Ошибка при обучении: {str(e)}")
            logger.error(f"Ошибка обучения: {e}", exc_info=True)


# ============= 4. РЕЗУЛЬТАТЫ И РЕКОМЕНДАЦИИ =============
if st.session_state.get('model_trained', False):
    st.header("📈 4. Результаты и рекомендации")

    env = st.session_state.env
    agent = st.session_state.agent
    tracker = st.session_state.tracker

    # Получить итоговую сводку
    final_summary = env.get_final_summary()

    # Метрики улучшения
    st.subheader("💰 Улучшение метрик")
    col1, col2, col3, col4 = st.columns(4)

    improvement = final_summary['improvement']
    col1.metric("Прирост прибыли", f"{improvement['profit']:,.0f} ₽", delta=f"{improvement['profit']:,.0f}")
    col2.metric("Прирост GMV", f"{improvement['gmv']:,.0f} ₽", delta=f"{improvement['gmv']:,.0f}")
    col3.metric("Изменение ROI", f"{improvement['roi']:.2f}%", delta=f"{improvement['roi']:.2f}%")
    col4.metric("Снижение OOS cost", f"{improvement['oos_cost_reduction']:,.0f} ₽")

    # Действия агента
    st.subheader("🎬 Распределение действий")
    col1, col2 = st.columns([1, 2])

    with col1:
        action_breakdown = final_summary['action_breakdown']
        st.write("**Статистика действий:**")
        for action, count in action_breakdown.items():
            st.write(f"- {action}: {count}")
        st.write(f"\n**Удалено SKU:** {final_summary['removed_skus']}")
        st.write(f"**Активных SKU:** {final_summary['active_skus']}")

    with col2:
        visualizer = AssortmentVisualizer(use_plotly=True)
        fig = visualizer.plot_action_distribution(action_breakdown)
        st.plotly_chart(fig, use_container_width=True)

    # Рекомендации
    st.subheader("📋 Рекомендации по SKU")
    recommendations_df = env.get_recommendations()

    # Фильтры
    col1, col2, col3 = st.columns(3)
    with col1:
        action_filter = st.multiselect(
            "Фильтр по действиям:",
            options=recommendations_df['Recommended_Action'].unique(),
            default=recommendations_df['Recommended_Action'].unique()
        )
    with col2:
        status_filter = st.multiselect(
            "Фильтр по статусу:",
            options=recommendations_df['Status'].unique(),
            default=recommendations_df['Status'].unique()
        )
    with col3:
        min_gmv = st.number_input("Минимальный GMV:", min_value=0.0, value=0.0)

    # Применить фильтры
    filtered_recs = recommendations_df[
        (recommendations_df['Recommended_Action'].isin(action_filter)) &
        (recommendations_df['Status'].isin(status_filter)) &
        (recommendations_df['Current_GMV'] >= min_gmv)
    ]

    st.dataframe(
        filtered_recs.style.background_gradient(subset=['Expected_Reward'], cmap='RdYlGn'),
        use_container_width=True
    )

    # Сохранить рекомендации
    st.session_state.recommendations = filtered_recs

    # Кнопка экспорта
    csv = filtered_recs.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Скачать рекомендации (CSV)",
        data=csv,
        file_name=f"sku_recommendations_{selected_store}.csv",
        mime="text/csv"
    )

    # Dashboard
    st.subheader("📊 Dashboard")
    dashboard_fig = visualizer.create_dashboard(
        initial_metrics=final_summary['initial_metrics'],
        final_metrics=final_summary['final_metrics'],
        action_breakdown=action_breakdown,
        learning_curve_data=tracker.get_learning_curve(),
        sku_df=st.session_state.sku_with_features
    )
    st.plotly_chart(dashboard_fig, use_container_width=True)


# ============= FOOTER =============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>SKU Optimization System v1.0.0 | Powered by Reinforcement Learning (DQN)</p>
    <p>⚠️ Рекомендации требуют валидации через A/B тестирование перед применением в production</p>
</div>
""", unsafe_allow_html=True)
