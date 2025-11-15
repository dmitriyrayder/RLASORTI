"""
RL Environment для оптимизации ассортимента SKU
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from enum import IntEnum
import logging

from .features import SKUFeatureEngineering, SegmentAnalyzer
from .metrics import BusinessMetricsCalculator, RewardFunction, AssortmentMetrics

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Действия агента"""
    KEEP = 0  # Оставить SKU как есть
    REMOVE = 1  # Убрать SKU из ассортимента
    INCREASE_DEPTH = 2  # Увеличить глубину стока (больше запасов)
    DECREASE_DEPTH = 3  # Уменьшить глубину стока


class SKUEnvironment:
    """
    Gym-like environment для управления ассортиментом SKU.

    State: вектор признаков текущего ассортимента и выбранного SKU
    Action: KEEP, REMOVE, INCREASE_DEPTH, DECREASE_DEPTH
    Reward: изменение прибыли - штрафы (OOS, inventory)
    """

    def __init__(
        self,
        sku_df: pd.DataFrame,
        feature_engineer: SKUFeatureEngineering,
        max_steps: int = 50,
        cannibalization_rate: float = 0.4,
        depth_change_factor: float = 0.15
    ):
        """
        Args:
            sku_df: DataFrame с агрегированными SKU (с признаками)
            feature_engineer: Объект для генерации признаков
            max_steps: Максимальное число шагов в эпизоде
            cannibalization_rate: Доля спроса, переключаемая на заменители при удалении SKU
            depth_change_factor: Множитель изменения стока при INCREASE/DECREASE_DEPTH
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
        self.current_sku_idx = 0
        self.active_skus = set(self.sku_df['Art'].tolist())
        self.removed_skus = set()

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

        Returns:
            Вектор начального состояния
        """
        self.sku_df = self.initial_sku_df.copy()
        self.active_skus = set(self.sku_df['Art'].tolist())
        self.removed_skus = set()
        self.current_step = 0
        self.current_sku_idx = 0
        self.action_history = []
        self.metrics_history = []

        self.current_metrics = self._calculate_current_metrics()
        self.initial_metrics = self.current_metrics

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Выполнить действие в среде.

        Args:
            action: Индекс действия (0-3)

        Returns:
            (next_state, reward, done, info)
        """
        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True, {'reason': 'max_steps_reached'}

        # Выбрать случайный активный SKU
        active_arts = list(self.active_skus)
        if not active_arts:
            return self._get_state(), 0.0, True, {'reason': 'no_active_skus'}

        art = active_arts[self.current_sku_idx % len(active_arts)]
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
        self.current_sku_idx = (self.current_sku_idx + 1) % max(1, len(active_arts))

        # Проверка завершения
        done = self.current_step >= self.max_steps

        # Информация
        info = {
            'art': art,
            'action': action_enum.name,
            'reward_components': reward_components,
            'metrics': self.current_metrics.to_dict(),
            'active_skus': len(self.active_skus),
            'removed_skus': len(self.removed_skus)
        }

        next_state = self._get_state()

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """
        Получить вектор состояния.

        State = признаки текущего ассортимента (агрегированные) +
                признаки выбранного SKU
        """
        # Признаки текущего ассортимента (агрегированные)
        active_df = self.sku_df[self.sku_df['Art'].isin(self.active_skus)]

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
        Применить действие к SKU.

        Args:
            art: Артикул SKU
            action: Действие

        Returns:
            Стоимость действия (action cost)
        """
        action_cost = 0.0

        if art not in self.active_skus:
            return 0.0  # SKU уже удален

        sku_row_idx = self.sku_df[self.sku_df['Art'] == art].index

        if len(sku_row_idx) == 0:
            return 0.0

        idx = sku_row_idx[0]

        if action == Action.KEEP:
            # Ничего не делаем
            pass

        elif action == Action.REMOVE:
            # Удаляем SKU
            self.active_skus.remove(art)
            self.removed_skus.add(art)

            # Перераспределяем спрос на заменители
            self._redistribute_demand(art)

            # Стоимость: потеря части спроса
            lost_gmv = self.sku_df.loc[idx, 'Sum_sum'] * (1 - self.cannibalization_rate)
            action_cost = lost_gmv * 0.1  # 10% от потерянного GMV

        elif action == Action.INCREASE_DEPTH:
            # Увеличиваем запасы
            stock_col = 'Stock' if 'Stock' in self.sku_df.columns else 'estimated_stock'
            old_stock = self.sku_df.loc[idx, stock_col]
            new_stock = old_stock * (1 + self.depth_change_factor)
            self.sku_df.loc[idx, stock_col] = new_stock

            # Пересчитываем stock coverage
            avg_daily = self.sku_df.loc[idx, 'avg_daily_qty']
            self.sku_df.loc[idx, 'stock_coverage_days'] = new_stock / max(0.1, avg_daily)

            # Стоимость: дополнительные инвестиции в запасы
            purchase_price = self.sku_df.loc[idx, 'purchase_price_mean']
            action_cost = (new_stock - old_stock) * purchase_price * 0.05  # 5% стоимости доп. запасов

        elif action == Action.DECREASE_DEPTH:
            # Уменьшаем запасы
            stock_col = 'Stock' if 'Stock' in self.sku_df.columns else 'estimated_stock'
            old_stock = self.sku_df.loc[idx, stock_col]
            new_stock = old_stock * (1 - self.depth_change_factor)
            self.sku_df.loc[idx, stock_col] = new_stock

            # Пересчитываем stock coverage
            avg_daily = self.sku_df.loc[idx, 'avg_daily_qty']
            self.sku_df.loc[idx, 'stock_coverage_days'] = new_stock / max(0.1, avg_daily)

            # Стоимость: риск OOS
            if new_stock < avg_daily * 3:  # Меньше 3 дней покрытия
                action_cost = self.sku_df.loc[idx, 'profit_per_day'] * 0.2  # Потенциальная потеря

        return action_cost

    def _redistribute_demand(self, removed_art: str):
        """
        Перераспределить спрос при удалении SKU на заменители.

        Args:
            removed_art: Артикул удаленного SKU
        """
        # Получаем объем удаляемого SKU
        removed_row = self.sku_df[self.sku_df['Art'] == removed_art]
        if removed_row.empty:
            return

        removed_qty = removed_row.iloc[0]['Qty_sum']
        removed_gmv = removed_row.iloc[0]['Sum_sum']

        # Находим заменители
        substitutes_allocation = self.segment_analyzer.estimate_cannibalization_effect(removed_art)

        # Распределяем спрос
        for sub_art, allocation_pct in substitutes_allocation.items():
            if sub_art in self.active_skus:
                sub_idx = self.sku_df[self.sku_df['Art'] == sub_art].index
                if len(sub_idx) > 0:
                    idx = sub_idx[0]
                    # Добавляем долю спроса
                    additional_qty = removed_qty * allocation_pct * self.cannibalization_rate
                    additional_gmv = removed_gmv * allocation_pct * self.cannibalization_rate

                    self.sku_df.loc[idx, 'Qty_sum'] += additional_qty
                    self.sku_df.loc[idx, 'Sum_sum'] += additional_gmv

                    # Обновляем производные метрики
                    self.sku_df.loc[idx, 'turnover'] = (
                        self.sku_df.loc[idx, 'Qty_sum'] /
                        max(1, self.sku_df.loc[idx, 'num_transactions'])
                    )
                    self.sku_df.loc[idx, 'avg_daily_qty'] = (
                        self.sku_df.loc[idx, 'Qty_sum'] /
                        max(1, self.sku_df.loc[idx, 'days_active'])
                    )

    def _calculate_current_metrics(self) -> AssortmentMetrics:
        """Вычислить метрики текущего ассортимента"""
        active_df = self.sku_df[self.sku_df['Art'].isin(self.active_skus)]
        return self.metrics_calc.calculate_metrics(active_df)

    def get_final_summary(self) -> Dict:
        """
        Получить итоговую сводку по эпизоду.

        Returns:
            Словарь с метриками до/после и историей
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
            'active_skus': len(self.active_skus),
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

        Returns:
            DataFrame с рекомендациями по SKU
        """
        recommendations = []

        for art, action, reward in self.action_history:
            sku_row = self.initial_sku_df[self.initial_sku_df['Art'] == art]
            if not sku_row.empty:
                row = sku_row.iloc[0]
                segment_col = 'Segment_<lambda>' if 'Segment_<lambda>' in row.index else 'Segment'

                recommendations.append({
                    'Art': art,
                    'Segment': row.get(segment_col, 'Unknown'),
                    'Current_GMV': row.get('Sum_sum', 0),
                    'Current_Profit': row.get('total_profit', 0),
                    'Recommended_Action': action.name,
                    'Expected_Reward': reward,
                    'Status': 'Removed' if art in self.removed_skus else 'Active'
                })

        return pd.DataFrame(recommendations).sort_values('Expected_Reward', ascending=False)
