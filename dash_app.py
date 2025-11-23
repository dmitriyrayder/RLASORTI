"""
Dash интерфейс для SKU Optimization System
Улучшенная версия с современным дизайном и функционалом
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import base64
import io
import logging

from sku_data_loader import SalesDataLoader
from sku_features import SKUFeatureEngineering, SegmentAnalyzer
from sku_environment import SKUEnvironment
from sku_agents import DQNAgent
from sku_metrics import PerformanceTracker

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Dash приложения с Bootstrap темой
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)

app.title = "SKU Optimization System - RL"

# Глобальное хранилище данных
class AppState:
    def __init__(self):
        self.data_loader = None
        self.df = None
        self.agent = None
        self.env = None
        self.tracker = None
        self.feature_eng = None
        self.sku_with_features = None
        self.training_complete = False

app_state = AppState()

# =========================
# СТИЛИ
# =========================
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "overflow-y": "auto"
}

CONTENT_STYLE = {
    "margin-left": "22rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

CARD_STYLE = {
    "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
    "border-radius": "10px",
    "margin-bottom": "20px"
}

# =========================
# SIDEBAR
# =========================
sidebar = html.Div(
    [
        html.H2("⚙️ Настройки", className="text-center mb-4"),
        html.Hr(),

        html.H5("RL Параметры", className="mt-4"),

        html.Label("Количество эпизодов:"),
        dcc.Slider(
            id='n-episodes',
            min=10,
            max=500,
            step=10,
            value=100,
            marks={10: '10', 100: '100', 300: '300', 500: '500'},
            tooltip={"placement": "bottom", "always_visible": True}
        ),

        html.Label("Макс. шагов на эпизод:", className="mt-3"),
        dcc.Slider(
            id='max-steps',
            min=10,
            max=100,
            step=5,
            value=50,
            marks={10: '10', 50: '50', 100: '100'},
            tooltip={"placement": "bottom", "always_visible": True}
        ),

        html.Label("Learning Rate:", className="mt-3"),
        dcc.Input(
            id='learning-rate',
            type='number',
            value=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            className="form-control"
        ),

        html.Label("Gamma (дисконт):", className="mt-3"),
        dcc.Slider(
            id='gamma',
            min=0.90,
            max=0.99,
            step=0.01,
            value=0.95,
            marks={0.90: '0.90', 0.95: '0.95', 0.99: '0.99'},
            tooltip={"placement": "bottom", "always_visible": True}
        ),

        html.Label("Epsilon (начальный):", className="mt-3"),
        dcc.Slider(
            id='epsilon-start',
            min=0.5,
            max=1.0,
            step=0.05,
            value=1.0,
            marks={0.5: '0.5', 0.75: '0.75', 1.0: '1.0'},
            tooltip={"placement": "bottom", "always_visible": True}
        ),

        html.Label("Epsilon (минимальный):", className="mt-3"),
        dcc.Slider(
            id='epsilon-min',
            min=0.01,
            max=0.2,
            step=0.01,
            value=0.05,
            marks={0.01: '0.01', 0.1: '0.1', 0.2: '0.2'},
            tooltip={"placement": "bottom", "always_visible": True}
        ),

        html.Hr(className="mt-4"),
        html.P(
            "SKU Optimization System v2.0",
            className="text-muted text-center small"
        ),
    ],
    style=SIDEBAR_STYLE,
)

# =========================
# ГЛАВНАЯ СТРАНИЦА
# =========================
content = html.Div(
    [
        # Заголовок
        dbc.Row([
            dbc.Col([
                html.H1("🎯 SKU Optimization System", className="text-primary mb-3"),
                html.P(
                    "Reinforcement Learning для умного управления ассортиментом",
                    className="lead text-muted"
                ),
            ], width=12)
        ]),

        html.Hr(),

        # Секция 1: Загрузка данных
        dbc.Card([
            dbc.CardHeader(html.H3("📂 1. Загрузка данных", className="text-white"), style={"background-color": "#007bff"}),
            dbc.CardBody([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt fa-3x mb-3"),
                        html.H5('Перетащите файл или нажмите для выбора'),
                        html.P('Поддерживаются форматы: XLSX, XLS, CSV', className="text-muted")
                    ]),
                    style={
                        'width': '100%',
                        'height': '150px',
                        'lineHeight': '150px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '10px',
                        'textAlign': 'center',
                        'background-color': '#f8f9fa'
                    },
                    multiple=False
                ),
                html.Div(id='upload-status', className="mt-3"),
                html.Div(id='data-summary', className="mt-3"),
            ])
        ], style=CARD_STYLE, className="mb-4"),

        # Секция 2: Выбор магазина
        html.Div(id='store-selection-section', children=[], className="mb-4"),

        # Секция 3: Обучение модели
        html.Div(id='training-section', children=[], className="mb-4"),

        # Секция 4: Результаты
        html.Div(id='results-section', children=[], className="mb-4"),

        # Секция 5: Dashboard
        html.Div(id='dashboard-section', children=[], className="mb-4"),

        # Хранилища данных
        dcc.Store(id='data-store'),
        dcc.Store(id='training-store'),
        dcc.Interval(id='training-interval', interval=1000, disabled=True),
    ],
    style=CONTENT_STYLE
)

app.layout = html.Div([sidebar, content])

# =========================
# CALLBACKS
# =========================

@app.callback(
    [Output('upload-status', 'children'),
     Output('data-summary', 'children'),
     Output('store-selection-section', 'children'),
     Output('data-store', 'data')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def upload_file(contents, filename):
    if contents is None:
        return "", "", [], None

    try:
        # Декодирование файла
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Сохранение временного файла
        temp_path = Path(f"temp_{filename}")
        with open(temp_path, 'wb') as f:
            f.write(decoded)

        # Загрузка данных
        data_loader = SalesDataLoader()
        df = data_loader.load(str(temp_path))

        # Удаление временного файла
        temp_path.unlink()

        # Сохранение в глобальное состояние
        app_state.data_loader = data_loader
        app_state.df = df

        # Статистика
        stats = data_loader.get_summary_stats()

        summary = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['total_records']:,}", className="text-primary"),
                        html.P("Записей", className="text-muted mb-0")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['unique_skus']:,}", className="text-success"),
                        html.P("SKU", className="text-muted mb-0")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['unique_stores']:,}", className="text-info"),
                        html.P("Магазинов", className="text-muted mb-0")
                    ])
                ], className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['total_gmv']:,.0f} ₽", className="text-warning"),
                        html.P("GMV", className="text-muted mb-0")
                    ])
                ], className="text-center")
            ], width=3),
        ])

        # Выбор магазина
        store_section = dbc.Card([
            dbc.CardHeader(html.H3("🏪 2. Выбор магазина", className="text-white"), style={"background-color": "#28a745"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Выберите магазин для оптимизации:", className="font-weight-bold"),
                        dcc.Dropdown(
                            id='store-dropdown',
                            options=[{'label': store, 'value': store} for store in stats['stores']],
                            value=stats['stores'][0] if stats['stores'] else None,
                            className="mb-3"
                        ),
                        html.Div(id='store-info')
                    ], width=12)
                ])
            ])
        ], style=CARD_STYLE)

        status = dbc.Alert(
            f"✅ Успешно загружено: {filename}",
            color="success",
            dismissable=True
        )

        return status, summary, store_section, {'loaded': True}

    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}", exc_info=True)
        return dbc.Alert(f"❌ Ошибка: {str(e)}", color="danger"), "", [], None


@app.callback(
    [Output('store-info', 'children'),
     Output('training-section', 'children')],
    Input('store-dropdown', 'value'),
    Input('data-store', 'data')
)
def update_store_info(selected_store, data_stored):
    if not data_stored or not selected_store or app_state.data_loader is None:
        return "", ""

    try:
        sku_agg = app_state.data_loader.get_sku_aggregates(selected_store)

        info = dbc.Alert([
            html.H5(f"📦 Магазин: {selected_store}", className="alert-heading"),
            html.Hr(),
            html.P(f"Доступно {len(sku_agg)} SKU для оптимизации"),
            html.P(f"Сегментов: {sku_agg['Segment'].nunique()}"),
            html.P(f"Общий GMV: {sku_agg['Sum_sum'].sum():,.0f} ₽"),
        ], color="info")

        # Секция обучения
        training_section = dbc.Card([
            dbc.CardHeader(html.H3("🧠 3. Обучение RL модели", className="text-white"), style={"background-color": "#ffc107"}),
            dbc.CardBody([
                dbc.Button(
                    [html.I(className="fas fa-play mr-2"), "Запустить обучение DQN агента"],
                    id='start-training-btn',
                    color="primary",
                    size="lg",
                    className="mb-3"
                ),
                html.Div(id='training-progress'),
                html.Div(id='training-results')
            ])
        ], style=CARD_STYLE)

        return info, training_section

    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        return dbc.Alert(f"❌ Ошибка: {str(e)}", color="danger"), ""


@app.callback(
    [Output('training-progress', 'children'),
     Output('training-results', 'children'),
     Output('results-section', 'children'),
     Output('dashboard-section', 'children')],
    Input('start-training-btn', 'n_clicks'),
    State('store-dropdown', 'value'),
    State('n-episodes', 'value'),
    State('max-steps', 'value'),
    State('learning-rate', 'value'),
    State('gamma', 'value'),
    State('epsilon-start', 'value'),
    State('epsilon-min', 'value'),
    prevent_initial_call=True
)
def train_model(n_clicks, selected_store, n_episodes, max_steps, lr, gamma, eps_start, eps_min):
    if not n_clicks or app_state.data_loader is None:
        return "", "", "", ""

    try:
        # Прогресс бар
        progress = dbc.Progress(value=0, id='training-progress-bar', className="mb-3")
        status = html.Div(id='training-status')

        # Загрузка данных
        sku_agg = app_state.data_loader.get_sku_aggregates(selected_store)

        # Feature engineering
        feature_eng = SKUFeatureEngineering(scaler_type='robust')
        sku_with_features = feature_eng.engineer_features(sku_agg, fit=True)

        # Environment
        env = SKUEnvironment(
            sku_df=sku_with_features,
            feature_engineer=feature_eng,
            max_steps=max_steps
        )

        # Agent
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            learning_rate=lr,
            gamma=gamma,
            epsilon=eps_start,
            epsilon_min=eps_min,
            epsilon_decay=0.995,
            buffer_size=5000,
            batch_size=64
        )

        # Tracker
        tracker = PerformanceTracker()

        # Обучение
        training_log = []
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            episode_losses = []

            for step in range(max_steps):
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                train_metrics = agent.train_step(state, action, reward, next_state, done)
                if train_metrics['loss'] > 0:
                    episode_losses.append(train_metrics['loss'])

                state = next_state
                if done:
                    break

            agent.update_epsilon()
            final_metrics = env.current_metrics
            tracker.record_episode(episode_reward, final_metrics)

            if (episode + 1) % 20 == 0:
                training_log.append({
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'epsilon': agent.epsilon,
                    'loss': np.mean(episode_losses) if episode_losses else 0
                })

        # Сохранение результатов
        app_state.agent = agent
        app_state.env = env
        app_state.tracker = tracker
        app_state.feature_eng = feature_eng
        app_state.sku_with_features = sku_with_features
        app_state.training_complete = True

        # Результаты обучения
        summary = tracker.get_summary()

        training_results = dbc.Alert([
            html.H4("✅ Обучение завершено!", className="alert-heading"),
            html.Hr(),
            dbc.Row([
                dbc.Col([html.P(f"Всего эпизодов: {summary['total_episodes']}")], width=6),
                dbc.Col([html.P(f"Средний Reward: {summary['avg_reward']:.2f}")], width=6),
            ]),
            dbc.Row([
                dbc.Col([html.P(f"Лучший Reward: {summary['best_reward']:.2f}")], width=6),
                dbc.Col([html.P(f"Последние 10 эп.: {summary['last_10_avg_reward']:.2f}")], width=6),
            ]),
        ], color="success")

        # График обучения
        episodes, rewards = tracker.get_learning_curve()
        fig_learning = go.Figure()
        fig_learning.add_trace(go.Scatter(
            x=episodes,
            y=rewards,
            mode='lines',
            name='Reward',
            line=dict(color='#007bff', width=2)
        ))
        fig_learning.update_layout(
            title="Learning Curve - DQN Agent",
            xaxis_title="Episode",
            yaxis_title="Reward (smoothed)",
            template="plotly_white",
            height=400
        )

        # Секция результатов
        final_summary = env.get_final_summary()
        improvement = final_summary['improvement']

        results_section = dbc.Card([
            dbc.CardHeader(html.H3("📈 4. Результаты и рекомендации", className="text-white"), style={"background-color": "#17a2b8"}),
            dbc.CardBody([
                html.H5("💰 Улучшение метрик", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{improvement['profit']:,.0f} ₽", className="text-success"),
                                html.P("Прирост прибыли", className="text-muted mb-0")
                            ])
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{improvement['gmv']:,.0f} ₽", className="text-info"),
                                html.P("Прирост GMV", className="text-muted mb-0")
                            ])
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{improvement['roi']:.2f}%", className="text-primary"),
                                html.P("Изменение ROI", className="text-muted mb-0")
                            ])
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{improvement['oos_cost_reduction']:,.0f} ₽", className="text-warning"),
                                html.P("Снижение OOS", className="text-muted mb-0")
                            ])
                        ], className="text-center")
                    ], width=3),
                ], className="mb-4"),

                html.H5("📋 Рекомендации по SKU", className="mb-3"),
                html.Div(id='recommendations-table'),

                dbc.Button(
                    [html.I(className="fas fa-download mr-2"), "Скачать рекомендации (CSV)"],
                    id='download-btn',
                    color="success",
                    className="mt-3"
                ),
                dcc.Download(id="download-recommendations")
            ])
        ], style=CARD_STYLE)

        # Таблица рекомендаций
        recommendations_df = env.get_recommendations()
        recommendations_table = dash_table.DataTable(
            data=recommendations_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in recommendations_df.columns],
            page_size=20,
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'font-family': 'Arial'
            },
            style_header={
                'backgroundColor': '#007bff',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Status', 'filter_query': '{Status} = "Removed"'},
                    'backgroundColor': '#ffebee',
                    'color': '#c62828'
                },
                {
                    'if': {'column_id': 'Status', 'filter_query': '{Status} = "Active"'},
                    'backgroundColor': '#e8f5e9',
                    'color': '#2e7d32'
                }
            ]
        )

        # Dashboard
        action_breakdown = final_summary['action_breakdown']

        fig_actions = go.Figure(data=[go.Pie(
            labels=list(action_breakdown.keys()),
            values=list(action_breakdown.values()),
            hole=0.4,
            marker=dict(colors=['#28a745', '#dc3545', '#007bff', '#ffc107'])
        )])
        fig_actions.update_layout(
            title="Распределение действий агента",
            template="plotly_white",
            height=400
        )

        dashboard = dbc.Card([
            dbc.CardHeader(html.H3("📊 5. Dashboard", className="text-white"), style={"background-color": "#6c757d"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=fig_learning)], width=6),
                    dbc.Col([dcc.Graph(figure=fig_actions)], width=6),
                ])
            ])
        ], style=CARD_STYLE)

        return progress, training_results, results_section, dashboard

    except Exception as e:
        logger.error(f"Ошибка обучения: {e}", exc_info=True)
        return "", dbc.Alert(f"❌ Ошибка: {str(e)}", color="danger"), "", ""


@app.callback(
    Output('recommendations-table', 'children'),
    Input('results-section', 'children')
)
def update_recommendations_table(results_content):
    if not app_state.training_complete or app_state.env is None:
        return ""

    recommendations_df = app_state.env.get_recommendations()

    table = dash_table.DataTable(
        data=recommendations_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in recommendations_df.columns],
        page_size=20,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'font-family': 'Arial'
        },
        style_header={
            'backgroundColor': '#007bff',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'Status', 'filter_query': '{Status} = "Removed"'},
                'backgroundColor': '#ffebee',
                'color': '#c62828'
            },
            {
                'if': {'column_id': 'Status', 'filter_query': '{Status} = "Active"'},
                'backgroundColor': '#e8f5e9',
                'color': '#2e7d32'
            }
        ],
        filter_action="native",
        sort_action="native"
    )

    return table


@app.callback(
    Output("download-recommendations", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_recommendations(n_clicks):
    if not app_state.training_complete or app_state.env is None:
        return None

    recommendations_df = app_state.env.get_recommendations()
    return dcc.send_data_frame(recommendations_df.to_csv, "sku_recommendations.csv", index=False)


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
