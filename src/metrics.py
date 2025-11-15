"""
Бизнес-метрики и reward функции для RL оптимизации SKU
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AssortmentMetrics:
    """Метрики ассортимента"""
    total_gmv: float
    total_profit: float
    total_qty: float
    avg_margin_pct: float
    num_skus: int
    num_segments: int
    out_of_stock_cost: float
    roi: float
    inventory_cost: float

    def to_dict(self) -> Dict:
        return {
            'GMV': self.total_gmv,
            'Profit': self.total_profit,
            'Quantity': self.total_qty,
            'AvgMargin%': self.avg_margin_pct,
            'NumSKUs': self.num_skus,
            'NumSegments': self.num_segments,
            'OOS_Cost': self.out_of_stock_cost,
            'ROI%': self.roi,
            'InventoryCost': self.inventory_cost
        }


class BusinessMetricsCalculator:
    """
    Калькулятор бизнес-метрик для системы оптимизации ассортимента.

    Метрики:
    - GMV (Gross Merchandise Value)
    - Profit (валовая прибыль)
    - ROI (Return on Investment)
    - Out-of-Stock cost
    - Inventory holding cost
    - Turnover rate
    """

    def __init__(self, inventory_holding_rate: float = 0.15):
        """
        Args:
            inventory_holding_rate: Годовая стоимость хранения запасов (% от закупочной цены)
        """
        self.inventory_holding_rate = inventory_holding_rate

    def calculate_metrics(
        self,
        sku_df: pd.DataFrame,
        time_period_days: int = 30
    ) -> AssortmentMetrics:
        """
        Расчет метрик для текущего ассортимента.

        Args:
            sku_df: DataFrame со SKU и их характеристиками
            time_period_days: Период времени для расчета (дней)

        Returns:
            AssortmentMetrics объект
        """
        # GMV
        total_gmv = sku_df['Sum_sum'].sum() if 'Sum_sum' in sku_df.columns else 0

        # Profit
        if 'total_profit' in sku_df.columns:
            total_profit = sku_df['total_profit'].sum()
        else:
            total_profit = (sku_df['margin_mean'] * sku_df['Qty_sum']).sum()

        # Quantity
        total_qty = sku_df['Qty_sum'].sum() if 'Qty_sum' in sku_df.columns else 0

        # Average margin %
        avg_margin_pct = sku_df['margin_pct_mean'].mean() if 'margin_pct_mean' in sku_df.columns else 0

        # Number of SKUs and segments
        num_skus = len(sku_df)
        segment_col = 'Segment_<lambda>' if 'Segment_<lambda>' in sku_df.columns else 'Segment'
        num_segments = sku_df[segment_col].nunique() if segment_col in sku_df.columns else 0

        # Out-of-stock cost (если есть)
        oos_cost = self._estimate_oos_cost(sku_df)

        # ROI
        total_investment = (sku_df['purchase_price_mean'] * sku_df['Qty_sum']).sum()
        roi = (total_profit / total_investment * 100) if total_investment > 0 else 0

        # Inventory cost
        inventory_cost = self._calculate_inventory_cost(sku_df, time_period_days)

        return AssortmentMetrics(
            total_gmv=total_gmv,
            total_profit=total_profit,
            total_qty=total_qty,
            avg_margin_pct=avg_margin_pct,
            num_skus=num_skus,
            num_segments=num_segments,
            out_of_stock_cost=oos_cost,
            roi=roi,
            inventory_cost=inventory_cost
        )

    def _estimate_oos_cost(self, sku_df: pd.DataFrame) -> float:
        """
        Оценка стоимости out-of-stock ситуаций.

        Эвристика: если stock coverage < 3 дней, то теряем потенциальные продажи
        """
        if 'stock_coverage_days' not in sku_df.columns:
            return 0.0

        # SKU с низким покрытием
        low_stock = sku_df[sku_df['stock_coverage_days'] < 3].copy()

        if low_stock.empty:
            return 0.0

        # Оцениваем упущенную прибыль
        lost_sales_days = 3 - low_stock['stock_coverage_days']
        lost_profit = (low_stock['profit_per_day'] * lost_sales_days).sum()

        return max(0, lost_profit)

    def _calculate_inventory_cost(self, sku_df: pd.DataFrame, time_period_days: int) -> float:
        """Стоимость хранения запасов"""
        if 'estimated_stock' not in sku_df.columns and 'Stock' not in sku_df.columns:
            return 0.0

        stock_col = 'Stock' if 'Stock' in sku_df.columns else 'estimated_stock'
        inventory_value = (sku_df[stock_col] * sku_df['purchase_price_mean']).sum()

        # Пропорциональная стоимость за период
        daily_rate = self.inventory_holding_rate / 365
        cost = inventory_value * daily_rate * time_period_days

        return cost


class RewardFunction:
    """
    Функция вознаграждения для RL агента.

    Компоненты reward:
    1. Прирост прибыли (основной компонент)
    2. Штраф за out-of-stock
    3. Штраф за избыток запасов
    4. Штраф за снижение качества ассортимента
    """

    def __init__(
        self,
        profit_weight: float = 1.0,
        oos_penalty_weight: float = 0.3,
        inventory_penalty_weight: float = 0.1,
        diversity_weight: float = 0.2
    ):
        """
        Args:
            profit_weight: Вес компонента прибыли
            oos_penalty_weight: Вес штрафа за out-of-stock
            inventory_penalty_weight: Вес штрафа за избыток запасов
            diversity_weight: Вес компонента разнообразия ассортимента
        """
        self.profit_weight = profit_weight
        self.oos_penalty_weight = oos_penalty_weight
        self.inventory_penalty_weight = inventory_penalty_weight
        self.diversity_weight = diversity_weight
        self.metrics_calc = BusinessMetricsCalculator()

    def calculate(
        self,
        prev_metrics: AssortmentMetrics,
        new_metrics: AssortmentMetrics,
        action_cost: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Вычисление reward.

        Args:
            prev_metrics: Метрики до действия
            new_metrics: Метрики после действия
            action_cost: Стоимость действия (переключение SKU и т.д.)

        Returns:
            (total_reward, components) где components - словарь компонентов reward
        """
        # 1. Profit component
        profit_delta = new_metrics.total_profit - prev_metrics.total_profit
        profit_reward = profit_delta * self.profit_weight

        # 2. Out-of-stock penalty
        oos_delta = new_metrics.out_of_stock_cost - prev_metrics.out_of_stock_cost
        oos_penalty = -oos_delta * self.oos_penalty_weight

        # 3. Inventory cost penalty
        inventory_delta = new_metrics.inventory_cost - prev_metrics.inventory_cost
        inventory_penalty = -inventory_delta * self.inventory_penalty_weight

        # 4. Diversity reward (поощряем разнообразие сегментов)
        diversity_delta = new_metrics.num_segments - prev_metrics.num_segments
        diversity_reward = diversity_delta * self.diversity_weight * 100  # scale up

        # 5. Action cost
        action_penalty = -action_cost

        # Total reward
        total_reward = (
            profit_reward +
            oos_penalty +
            inventory_penalty +
            diversity_reward +
            action_penalty
        )

        components = {
            'profit': profit_reward,
            'oos_penalty': oos_penalty,
            'inventory_penalty': inventory_penalty,
            'diversity': diversity_reward,
            'action_cost': action_penalty,
            'total': total_reward
        }

        return total_reward, components

    def calculate_simple(
        self,
        prev_profit: float,
        new_profit: float,
        prev_oos_cost: float,
        new_oos_cost: float
    ) -> float:
        """
        Упрощенная версия reward (только прибыль и OOS).

        Args:
            prev_profit: Прибыль до
            new_profit: Прибыль после
            prev_oos_cost: OOS cost до
            new_oos_cost: OOS cost после

        Returns:
            Reward value
        """
        profit_delta = new_profit - prev_profit
        oos_delta = new_oos_cost - prev_oos_cost

        reward = (
            profit_delta * self.profit_weight -
            oos_delta * self.oos_penalty_weight
        )

        return reward


class PerformanceTracker:
    """
    Отслеживание производительности системы во время обучения и эксплуатации.
    """

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_metrics: List[Dict] = []
        self.best_reward: float = -np.inf
        self.best_metrics: Optional[AssortmentMetrics] = None

    def record_episode(self, total_reward: float, metrics: AssortmentMetrics):
        """Записать результаты эпизода"""
        self.episode_rewards.append(total_reward)
        self.episode_metrics.append(metrics.to_dict())

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_metrics = metrics

    def get_summary(self) -> Dict:
        """Получить сводку по обучению"""
        if not self.episode_rewards:
            return {}

        return {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'best_reward': self.best_reward,
            'last_10_avg_reward': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards),
            'reward_std': np.std(self.episode_rewards),
            'best_metrics': self.best_metrics.to_dict() if self.best_metrics else None
        }

    def get_learning_curve(self) -> Tuple[List[float], List[float]]:
        """
        Получить кривую обучения (сглаженная).

        Returns:
            (episodes, smoothed_rewards)
        """
        if not self.episode_rewards:
            return [], []

        # Moving average с окном 10
        window = min(10, len(self.episode_rewards))
        smoothed = np.convolve(
            self.episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )

        episodes = list(range(window - 1, len(self.episode_rewards)))
        return episodes, smoothed.tolist()

    def reset(self):
        """Сброс трекера"""
        self.episode_rewards = []
        self.episode_metrics = []
        self.best_reward = -np.inf
        self.best_metrics = None
