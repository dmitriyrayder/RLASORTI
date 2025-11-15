"""
Модуль визуализации результатов оптимизации ассортимента
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class AssortmentVisualizer:
    """
    Класс для визуализации результатов оптимизации ассортимента.
    """

    def __init__(self, use_plotly: bool = True):
        """
        Args:
            use_plotly: Использовать Plotly (интерактивные графики) вместо Matplotlib
        """
        self.use_plotly = use_plotly

    def plot_learning_curve(
        self,
        episodes: List[int],
        rewards: List[float],
        title: str = "Learning Curve"
    ):
        """
        График кривой обучения.

        Args:
            episodes: Номера эпизодов
            rewards: Награды по эпизодам
            title: Заголовок графика
        """
        if self.use_plotly:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=episodes,
                y=rewards,
                mode='lines',
                name='Reward',
                line=dict(color='blue', width=2)
            ))

            # Moving average
            if len(rewards) > 10:
                window = min(10, len(rewards))
                ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
                fig.add_trace(go.Scatter(
                    x=episodes[window-1:],
                    y=ma,
                    mode='lines',
                    name='MA(10)',
                    line=dict(color='red', width=2, dash='dash')
                ))

            fig.update_layout(
                title=title,
                xaxis_title="Episode",
                yaxis_title="Total Reward",
                hovermode='x unified'
            )
            return fig
        else:
            plt.figure(figsize=(12, 6))
            plt.plot(episodes, rewards, label='Reward', alpha=0.6)

            # Moving average
            if len(rewards) > 10:
                window = min(10, len(rewards))
                ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
                plt.plot(episodes[window-1:], ma, label='MA(10)', linewidth=2, color='red')

            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            return plt.gcf()

    def plot_metrics_comparison(
        self,
        initial_metrics: Dict,
        final_metrics: Dict,
        title: str = "Metrics Before vs After"
    ):
        """
        Сравнение метрик до и после оптимизации.

        Args:
            initial_metrics: Начальные метрики
            final_metrics: Финальные метрики
            title: Заголовок
        """
        metrics_to_plot = ['GMV', 'Profit', 'ROI%', 'NumSKUs']
        initial_values = [initial_metrics.get(m, 0) for m in metrics_to_plot]
        final_values = [final_metrics.get(m, 0) for m in metrics_to_plot]

        if self.use_plotly:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Before',
                x=metrics_to_plot,
                y=initial_values,
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='After',
                x=metrics_to_plot,
                y=final_values,
                marker_color='darkblue'
            ))

            fig.update_layout(
                title=title,
                xaxis_title="Metric",
                yaxis_title="Value",
                barmode='group'
            )
            return fig
        else:
            x = np.arange(len(metrics_to_plot))
            width = 0.35

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width/2, initial_values, width, label='Before', color='lightblue')
            ax.bar(x + width/2, final_values, width, label='After', color='darkblue')

            ax.set_xlabel('Metric')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_to_plot)
            ax.legend()
            plt.grid(True, alpha=0.3, axis='y')
            return fig

    def plot_action_distribution(
        self,
        action_breakdown: Dict[str, int],
        title: str = "Action Distribution"
    ):
        """
        Распределение действий агента.

        Args:
            action_breakdown: Словарь {action_name: count}
            title: Заголовок
        """
        actions = list(action_breakdown.keys())
        counts = list(action_breakdown.values())

        if self.use_plotly:
            fig = go.Figure(data=[go.Pie(
                labels=actions,
                values=counts,
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            fig.update_layout(title=title)
            return fig
        else:
            plt.figure(figsize=(8, 8))
            plt.pie(counts, labels=actions, autopct='%1.1f%%', startangle=90)
            plt.title(title)
            return plt.gcf()

    def plot_segment_analysis(
        self,
        sku_df: pd.DataFrame,
        segment_col: str = 'Segment_<lambda>',
        title: str = "Segment Analysis"
    ):
        """
        Анализ сегментов (GMV, количество SKU).

        Args:
            sku_df: DataFrame с SKU
            segment_col: Название колонки с сегментом
            title: Заголовок
        """
        if segment_col not in sku_df.columns:
            segment_col = 'Segment'

        segment_stats = sku_df.groupby(segment_col).agg({
            'Art': 'count',
            'Sum_sum': 'sum'
        }).rename(columns={'Art': 'SKU_count', 'Sum_sum': 'GMV'})

        segment_stats = segment_stats.sort_values('GMV', ascending=False).head(20)

        if self.use_plotly:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('GMV by Segment', 'SKU Count by Segment'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}]]
            )

            fig.add_trace(
                go.Bar(x=segment_stats.index, y=segment_stats['GMV'], name='GMV'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=segment_stats.index, y=segment_stats['SKU_count'], name='SKU Count'),
                row=1, col=2
            )

            fig.update_layout(title=title, showlegend=False, height=500)
            return fig
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            segment_stats['GMV'].plot(kind='bar', ax=axes[0], color='steelblue')
            axes[0].set_title('GMV by Segment')
            axes[0].set_ylabel('GMV')
            axes[0].tick_params(axis='x', rotation=45)

            segment_stats['SKU_count'].plot(kind='bar', ax=axes[1], color='coral')
            axes[1].set_title('SKU Count by Segment')
            axes[1].set_ylabel('Number of SKUs')
            axes[1].tick_params(axis='x', rotation=45)

            plt.suptitle(title)
            plt.tight_layout()
            return fig

    def plot_sku_ranking(
        self,
        sku_df: pd.DataFrame,
        top_n: int = 20,
        metric: str = 'Sum_sum',
        title: Optional[str] = None
    ):
        """
        Топ SKU по метрике.

        Args:
            sku_df: DataFrame с SKU
            top_n: Количество топовых SKU
            metric: Метрика для ранжирования
            title: Заголовок
        """
        if metric not in sku_df.columns:
            logger.warning(f"Метрика {metric} не найдена в DataFrame")
            return None

        top_skus = sku_df.nlargest(top_n, metric)[['Art', metric]]

        if title is None:
            title = f"Top {top_n} SKUs by {metric}"

        if self.use_plotly:
            fig = px.bar(
                top_skus,
                x='Art',
                y=metric,
                title=title,
                labels={'Art': 'SKU', metric: metric}
            )
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        else:
            plt.figure(figsize=(14, 6))
            plt.bar(top_skus['Art'], top_skus[metric], color='teal')
            plt.xlabel('SKU')
            plt.ylabel(metric)
            plt.title(title)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return plt.gcf()

    def plot_profit_vs_turnover(
        self,
        sku_df: pd.DataFrame,
        title: str = "Profit vs Turnover (SKU Scatter)"
    ):
        """
        Scatter plot: прибыль vs оборачиваемость.

        Args:
            sku_df: DataFrame с SKU
            title: Заголовок
        """
        segment_col = 'Segment_<lambda>' if 'Segment_<lambda>' in sku_df.columns else 'Segment'

        if self.use_plotly:
            fig = px.scatter(
                sku_df,
                x='turnover',
                y='total_profit',
                color=segment_col,
                size='Sum_sum',
                hover_data=['Art'],
                title=title,
                labels={
                    'turnover': 'Turnover (Qty/Transaction)',
                    'total_profit': 'Total Profit',
                    segment_col: 'Segment'
                }
            )
            return fig
        else:
            plt.figure(figsize=(12, 8))
            for segment in sku_df[segment_col].unique():
                segment_data = sku_df[sku_df[segment_col] == segment]
                plt.scatter(
                    segment_data['turnover'],
                    segment_data['total_profit'],
                    label=segment,
                    alpha=0.6,
                    s=100
                )

            plt.xlabel('Turnover (Qty/Transaction)')
            plt.ylabel('Total Profit')
            plt.title(title)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            return plt.gcf()

    def create_dashboard(
        self,
        initial_metrics: Dict,
        final_metrics: Dict,
        action_breakdown: Dict,
        learning_curve_data: Tuple[List, List],
        sku_df: pd.DataFrame
    ):
        """
        Создать полный dashboard с несколькими графиками.

        Args:
            initial_metrics: Начальные метрики
            final_metrics: Финальные метрики
            action_breakdown: Распределение действий
            learning_curve_data: (episodes, rewards) для кривой обучения
            sku_df: DataFrame с SKU

        Returns:
            Plotly Figure (если use_plotly) или список Matplotlib figures
        """
        if not self.use_plotly:
            figs = [
                self.plot_metrics_comparison(initial_metrics, final_metrics),
                self.plot_learning_curve(*learning_curve_data),
                self.plot_action_distribution(action_breakdown),
                self.plot_segment_analysis(sku_df)
            ]
            return figs

        # Plotly dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Metrics Before vs After',
                'Learning Curve',
                'Action Distribution',
                'Segment GMV'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'pie'}, {'type': 'bar'}]
            ]
        )

        # 1. Metrics comparison
        metrics_to_plot = ['GMV', 'Profit', 'ROI%']
        initial_values = [initial_metrics.get(m, 0) for m in metrics_to_plot]
        final_values = [final_metrics.get(m, 0) for m in metrics_to_plot]

        fig.add_trace(
            go.Bar(name='Before', x=metrics_to_plot, y=initial_values, marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='After', x=metrics_to_plot, y=final_values, marker_color='darkblue'),
            row=1, col=1
        )

        # 2. Learning curve
        episodes, rewards = learning_curve_data
        fig.add_trace(
            go.Scatter(x=episodes, y=rewards, mode='lines', name='Reward'),
            row=1, col=2
        )

        # 3. Action distribution
        actions = list(action_breakdown.keys())
        counts = list(action_breakdown.values())
        fig.add_trace(
            go.Pie(labels=actions, values=counts, name='Actions'),
            row=2, col=1
        )

        # 4. Segment GMV
        segment_col = 'Segment_<lambda>' if 'Segment_<lambda>' in sku_df.columns else 'Segment'
        segment_gmv = sku_df.groupby(segment_col)['Sum_sum'].sum().nlargest(10)
        fig.add_trace(
            go.Bar(x=segment_gmv.index, y=segment_gmv.values, name='GMV'),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True, title_text="SKU Optimization Dashboard")
        return fig
