"""
RL Environment для оптимизации ассортимента SKU
Исправленная версия с корректной логикой
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from enum import IntEnum
import logging

from sku_features import SKUFeatureEngineering, SegmentAnalyzer
from sku_metrics import BusinessMetricsCalculator, RewardFunction, AssortmentMetrics

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Действия агента"""
    KEEP = 0
    REMOVE = 1
    INCREASE_DEPTH = 2
    DECREASE_DEPTH = 3


class SKUEnvironment:
    """
    Gym-like environment для управления ассортиментом SKU (ИСПРАВЛЕННЫЙ).
    """

    def __init__(
        self,
        sku_df: pd.DataFrame,
        feature_engineer: SKUFeatureEngineering,
        max_steps: int = 50,
        cannibalization_rate: float = 0.5,  # Увеличено с 0.4
        depth_change_factor: float = 0.15
    ):
        """
        Args:
            sku_df: DataFrame с агрегированными SKU (с признаками)
            feature_engineer: Объект для генерации признаков
            max_steps: Максимальное число шагов в эпизоде
            cannibalization_rate: Доля спроса, переключаемая на заменители
            depth_change_factor: Множитель изменения стока
        """
        self.initial_sku_df = sku_df.copy()
        self.sku_df = sku_df.copy()
        self.feature_engineer = feature_engineer
        self.max_steps = max_steps
        self.cannibalization_rate = cannibalization_rate
        self.depth_change_factor = depth_change_factor

        # Инициализация вспомогательных объектов
        self.segment_analyzer = SegmentAnalyzer(sku_df)
        self.metrics_calc = BusinessMetricsCalculator()
        self.reward_func = RewardFunction()

        # Состояние среды
        self.current_step = 0
        self.active_skus = list(self.sku_df['Art'].tolist())  # ИСПРАВЛЕНО: list вместо set
        self.removed_skus = set()

        # ИСПРАВЛЕНО: используем рандомизированную очередь вместо модуля
        self.sku_queue = []

        # История
        self.action_history: List[Tuple[str, Action, float]] = []
        self.metrics_history: List[AssortmentMetrics] = []

        # Текущие метрики
        self.current_metrics = self._calculate_current_metrics()
        self.initial_metrics = self.current_metrics

        logger.info(f"Environment инициализирован: {len(self.sku_df)} SKU, {max_steps} шагов")

    @property
    def state_dim(self) -> int:
        """Размерность state vector"""
        return self.feature_engineer.get_state_dim()

    @property
    def action_dim(self) -> int:
        """Количество возможных действий"""
        return len(Action)

    def reset(self) -> np.ndarray:
        """
        Сброс среды в начальное состояние.
        """
        self.sku_df = self.initial_sku_df.copy()
        self.active_skus = list(self.sku_df['Art'].tolist())
        self.removed_skus = set()
        self.current_step = 0
        self.action_history = []
        self.metrics_history = []

        # ИСПРАВЛЕНО: создаем новую рандомизированную очередь
        self.sku_queue = self.active_skus.copy()
        np.random.shuffle(self.sku_queue)

        self.current_metrics = self._calculate_current_metrics()
        self.initial_metrics = self.current_metrics

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Выполнить действие в среде (ИСПРАВЛЕНО).
        """
        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True, {'reason': 'max_steps_reached'}

        # ИСПРАВЛЕНО: берем следующий SKU из очереди
        if not self.sku_queue:
            # Если очередь закончилась, создаем новую
            if self.active_skus:
                self.sku_queue = [art for art in self.active_skus if art not in self.removed_skus]
                np.random.shuffle(self.sku_queue)
            else:
                return self._get_state(), 0.0, True, {'reason': 'no_active_skus'}

        art = self.sku_queue.pop(0)

        # Проверяем, что SKU все еще активен
        if art in self.removed_skus:
            return self.step(action)  # Рекурсивно берем следующий

        action_enum = Action(action)

        # Сохранить состояние до действия
        prev_metrics = self.current_metrics

        # Применить действие
        action_cost = self._apply_action(art, action_enum)

        # Вычислить новые метрики
        self.current_metrics = self._calculate_current_metrics()

        # Вычислить reward
        reward, reward_components = self.reward_func.calculate(
            prev_metrics,
            self.current_metrics,
            action_cost
        )

        # Записать историю
        self.action_history.append((art, action_enum, reward))
        self.metrics_history.append(self.current_metrics)

        # Обновить шаг
        self.current_step += 1

        # Проверка завершения
        done = self.current_step >= self.max_steps

        # Информация
        info = {
            'art': art,
            'action': action_enum.name,
            'reward_components': reward_components,
            'metrics': self.current_metrics.to_dict(),
            'active_skus': len([art for art in self.active_skus if art not in self.removed_skus]),
            'removed_skus': len(self.removed_skus)
        }

        next_state = self._get_state()

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """
        Получить вектор состояния (ИСПРАВЛЕНО).
        """
        # Признаки текущего ассортимента (агрегированные)
        active_arts = [art for art in self.active_skus if art not in self.removed_skus]
        active_df = self.sku_df[self.sku_df['Art'].isin(active_arts)]

        if active_df.empty:
            return np.zeros(self.state_dim, dtype=np.float32)

        # Берем средние значения по признакам
        agg_features = []
        for feat in self.feature_engineer.feature_names:
            if feat in active_df.columns:
                agg_features.append(active_df[feat].mean())
            else:
                agg_features.append(0.0)

        return np.array(agg_features, dtype=np.float32)

    def _apply_action(self, art: str, action: Action) -> float:
        """
        Применить действие к SKU (ИСПРАВЛЕНО: улучшенные costs).
        """
        action_cost = 0.0

        if art in self.removed_skus:
            return 0.0

        sku_row_idx = self.sku_df[self.sku_df['Art'] == art].index

        if len(sku_row_idx) == 0:
            return 0.0

        idx = sku_row_idx[0]

        if action == Action.KEEP:
            pass

        elif action == Action.REMOVE:
            # Удаляем SKU
            self.removed_skus.add(art)

            # Перераспределяем спрос на заменители
            self._redistribute_demand(art)

            # ИСПРАВЛЕНО: более реалистичная стоимость удаления
            lost_gmv = self.sku_df.loc[idx, 'Sum_sum'] * (1 - self.cannibalization_rate)
            lost_profit = self.sku_df.loc[idx, 'total_profit'] * (1 - self.cannibalization_rate)
            action_cost = lost_profit * 0.3  # 30% от потерянной прибыли

        elif action == Action.INCREASE_DEPTH:
            # Увеличиваем запасы
            stock_col = 'Stock' if 'Stock' in self.sku_df.columns else 'estimated_stock'
            old_stock = self.sku_df.loc[idx, stock_col]
            new_stock = old_stock * (1 + self.depth_change_factor)
            self.sku_df.loc[idx, stock_col] = new_stock

            # Пересчитываем stock coverage
            avg_daily = self.sku_df.loc[idx, 'avg_daily_qty']
            self.sku_df.loc[idx, 'stock_coverage_days'] = new_stock / max(0.1, avg_daily)

            # ИСПРАВЛЕНО: стоимость увеличения запасов
            purchase_price = self.sku_df.loc[idx, 'purchase_price_mean']
            additional_stock = new_stock - old_stock
            action_cost = additional_stock * purchase_price * 0.1  # 10% стоимости

        elif action == Action.DECREASE_DEPTH:
            # Уменьшаем запасы
            stock_col = 'Stock' if 'Stock' in self.sku_df.columns else 'estimated_stock'
            old_stock = self.sku_df.loc[idx, stock_col]
            new_stock = max(1, old_stock * (1 - self.depth_change_factor))  # Минимум 1
            self.sku_df.loc[idx, stock_col] = new_stock

            # Пересчитываем stock coverage
            avg_daily = self.sku_df.loc[idx, 'avg_daily_qty']
            self.sku_df.loc[idx, 'stock_coverage_days'] = new_stock / max(0.1, avg_daily)

            # ИСПРАВЛЕНО: риск OOS при низком стоке
            if new_stock < avg_daily * 5:
                profit_per_day = self.sku_df.loc[idx, 'profit_per_day']
                action_cost = profit_per_day * 0.5  # Потенциальная потеря

        return action_cost

    def _redistribute_demand(self, removed_art: str):
        """
        Перераспределить спрос при удалении SKU (ИСПРАВЛЕНО).
        """
        removed_row = self.sku_df[self.sku_df['Art'] == removed_art]
        if removed_row.empty:
            return

        removed_qty = removed_row.iloc[0]['Qty_sum']
        removed_gmv = removed_row.iloc[0]['Sum_sum']
        removed_profit = removed_row.iloc[0]['total_profit']

        # Находим заменители
        substitutes_allocation = self.segment_analyzer.estimate_cannibalization_effect(removed_art)

        if not substitutes_allocation:
            return

        # Распределяем спрос
        for sub_art, allocation_pct in substitutes_allocation.items():
            if sub_art not in self.removed_skus:
                sub_idx = self.sku_df[self.sku_df['Art'] == sub_art].index
                if len(sub_idx) > 0:
                    idx = sub_idx[0]

                    # Добавляем долю спроса
                    additional_qty = removed_qty * allocation_pct * self.cannibalization_rate
                    additional_gmv = removed_gmv * allocation_pct * self.cannibalization_rate
                    additional_profit = removed_profit * allocation_pct * self.cannibalization_rate

                    self.sku_df.loc[idx, 'Qty_sum'] += additional_qty
                    self.sku_df.loc[idx, 'Sum_sum'] += additional_gmv
                    self.sku_df.loc[idx, 'total_profit'] += additional_profit

                    # Обновляем производные метрики
                    num_trans = self.sku_df.loc[idx, 'num_transactions']
                    days_active = self.sku_df.loc[idx, 'days_active']

                    self.sku_df.loc[idx, 'turnover'] = self.sku_df.loc[idx, 'Qty_sum'] / max(1, num_trans)
                    self.sku_df.loc[idx, 'avg_daily_qty'] = self.sku_df.loc[idx, 'Qty_sum'] / max(1, days_active)
                    self.sku_df.loc[idx, 'profit_per_day'] = self.sku_df.loc[idx, 'total_profit'] / max(1, days_active)

    def _calculate_current_metrics(self) -> AssortmentMetrics:
        """Вычислить метрики текущего ассортимента"""
        active_arts = [art for art in self.active_skus if art not in self.removed_skus]
        active_df = self.sku_df[self.sku_df['Art'].isin(active_arts)]
        return self.metrics_calc.calculate_metrics(active_df)

    def get_final_summary(self) -> Dict:
        """
        Получить итоговую сводку по эпизоду.
        """
        return {
            'initial_metrics': self.initial_metrics.to_dict(),
            'final_metrics': self.current_metrics.to_dict(),
            'improvement': {
                'profit': self.current_metrics.total_profit - self.initial_metrics.total_profit,
                'gmv': self.current_metrics.total_gmv - self.initial_metrics.total_gmv,
                'roi': self.current_metrics.roi - self.initial_metrics.roi,
                'oos_cost_reduction': self.initial_metrics.out_of_stock_cost - self.current_metrics.out_of_stock_cost
            },
            'num_actions': len(self.action_history),
            'removed_skus': len(self.removed_skus),
            'active_skus': len([art for art in self.active_skus if art not in self.removed_skus]),
            'action_breakdown': self._get_action_breakdown()
        }

    def _get_action_breakdown(self) -> Dict[str, int]:
        """Разбивка по типам действий"""
        breakdown = {action.name: 0 for action in Action}
        for _, action, _ in self.action_history:
            breakdown[action.name] += 1
        return breakdown

    def get_recommendations(self) -> pd.DataFrame:
        """
        Получить рекомендации на основе истории действий.
        """
        recommendations = []

        for art, action, reward in self.action_history:
            sku_row = self.initial_sku_df[self.initial_sku_df['Art'] == art]
            if not sku_row.empty:
                row = sku_row.iloc[0]

                recommendations.append({
                    'Art': art,
                    'Segment': row.get('Segment', 'Unknown'),
                    'Current_GMV': row.get('Sum_sum', 0),
                    'Current_Profit': row.get('total_profit', 0),
                    'Recommended_Action': action.name,
                    'Expected_Reward': reward,
                    'Status': 'Removed' if art in self.removed_skus else 'Active'
                })

        return pd.DataFrame(recommendations).sort_values('Expected_Reward', ascending=False)
