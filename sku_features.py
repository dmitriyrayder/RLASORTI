"""
Feature Engineering для SKU оптимизации
Исправленная версия с корректными математическими формулами
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import RobustScaler, StandardScaler
import logging

logger = logging.getLogger(__name__)


class SKUFeatureEngineering:
    """
    Создание признаков для RL агента с исправленными формулами.
    """

    def __init__(self, scaler_type: str = 'robust'):
        """
        Args:
            scaler_type: 'standard' или 'robust' (устойчив к выбросам)
        """
        self.scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def engineer_features(self, sku_df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Создание признаков для SKU с исправленными формулами.
        """
        df = sku_df.copy()

        # 1. Turnover (оборачиваемость)
        df['turnover'] = df['Qty_sum'] / df['num_transactions'].clip(lower=1)

        # 2. Velocity (скорость продаж)
        df['velocity'] = df['avg_daily_qty']

        # 3. Stock coverage (ИСПРАВЛЕНО: более обоснованная эвристика)
        if 'Stock' in df.columns:
            df['stock_coverage_days'] = df['Stock'] / df['avg_daily_qty'].clip(lower=0.1)
        else:
            # Эвристика: 2x недельные продажи (более консервативно)
            df['estimated_stock'] = (df['avg_daily_qty'] * 7 * 2).clip(lower=1)
            df['stock_coverage_days'] = df['estimated_stock'] / df['avg_daily_qty'].clip(lower=0.1)

        # 4. Продуктовая давность
        current_date = df['last_sale_date'].max()
        df['days_since_last_sale'] = (current_date - df['last_sale_date']).dt.days

        # 5. Contribution to segment GMV (ИСПРАВЛЕНО: убрано 'Segment_<lambda>')
        segment_gmv = df.groupby('Segment')['Sum_sum'].transform('sum')
        df['segment_contribution_pct'] = (df['Sum_sum'] / segment_gmv * 100).fillna(0)

        # 6. Price positioning в сегменте (ИСПРАВЛЕНО)
        df['price_vs_segment_avg'] = df.groupby('Segment')['Price_mean'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

        # 7. Margin positioning (ИСПРАВЛЕНО)
        df['margin_vs_segment_avg'] = df.groupby('Segment')['margin_mean'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

        # 8. ABC классификация по GMV (ИСПРАВЛЕНО)
        df = df.join(
            df.groupby('Segment')['Sum_sum'].apply(self._abc_classification),
            on=['Art', 'Segment'],
            rsuffix='_abc'
        )
        if 'Sum_sum_abc' in df.columns:
            df['abc_class'] = df['Sum_sum_abc']
            df = df.drop('Sum_sum_abc', axis=1)
        else:
            df['abc_class'] = 'C'

        # 9. Частота продаж
        df['transaction_frequency'] = df['num_transactions'] / df['days_active'].clip(lower=1)

        # 10. Average basket size
        df['avg_basket_size'] = df['avg_transaction_qty']

        # 11. Profit per day
        df['profit_per_day'] = df['total_profit'] / df['days_active'].clip(lower=1)

        # 12. ROI
        df['roi'] = np.where(
            df['purchase_price_mean'] > 0,
            (df['margin_mean'] / df['purchase_price_mean']) * 100,
            0
        )

        # 13. Cannibalization risk
        df['segment_sku_count'] = df.groupby('Segment')['Art'].transform('count')
        df['cannibalization_risk'] = np.where(
            df['segment_sku_count'] > 10,
            np.log1p(df['segment_sku_count']) / 10,
            df['segment_sku_count'] / 10
        )

        # 14. Stability score (ИСПРАВЛЕНО: правильная формула на основе коэффициента вариации)
        # CV = std / mean, меньше CV = более стабильно
        # Нормализуем в диапазон [0, 1], где 1 = максимальная стабильность
        transaction_std = df.groupby('Art')['Qty_sum'].transform('std').fillna(0)
        transaction_mean = df['avg_transaction_qty'].clip(lower=0.1)
        cv = transaction_std / transaction_mean
        df['stability_score'] = 1 / (1 + cv)  # Преобразование в [0, 1]

        logger.info(f"Создано {len(df.columns) - len(sku_df.columns)} новых признаков")

        # Нормализация числовых признаков для RL
        numeric_features = [
            'turnover', 'velocity', 'stock_coverage_days', 'days_since_last_sale',
            'segment_contribution_pct', 'transaction_frequency', 'profit_per_day',
            'roi', 'cannibalization_risk', 'stability_score',
            'price_vs_segment_avg', 'margin_vs_segment_avg'
        ]

        # Заполнение NaN и inf
        for feat in numeric_features:
            df[feat] = df[feat].replace([np.inf, -np.inf], np.nan).fillna(0)

        if fit:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
            self.is_fitted = True
            self.feature_names = numeric_features
        elif self.is_fitted:
            df[numeric_features] = self.scaler.transform(df[numeric_features])

        return df

    @staticmethod
    def _abc_classification(series: pd.Series) -> pd.Series:
        """
        ABC классификация (Pareto):
        A: top 80% GMV
        B: next 15%
        C: last 5%
        """
        sorted_series = series.sort_values(ascending=False)
        cumsum = sorted_series.cumsum()
        total = sorted_series.sum()

        result = pd.Series('C', index=series.index)

        if total > 0:
            result.loc[cumsum <= 0.80 * total] = 'A'
            result.loc[(cumsum > 0.80 * total) & (cumsum <= 0.95 * total)] = 'B'

        return result

    def get_state_vector(self, sku_row: pd.Series) -> np.ndarray:
        """
        Получить вектор состояния для конкретного SKU.
        """
        if not self.is_fitted:
            raise ValueError("Feature engineering не обучен. Используйте engineer_features с fit=True")

        features = [sku_row.get(f, 0) for f in self.feature_names]
        return np.array(features, dtype=np.float32)

    def get_state_dim(self) -> int:
        """Размерность вектора состояния"""
        return len(self.feature_names) if self.feature_names else 12


class SegmentAnalyzer:
    """
    Анализ сегментов для понимания эффектов каннибализации.
    """

    def __init__(self, sku_df: pd.DataFrame):
        self.sku_df = sku_df
        self.segment_stats = self._compute_segment_stats()

    def _compute_segment_stats(self) -> pd.DataFrame:
        """Статистика по сегментам"""
        stats = self.sku_df.groupby('Segment').agg({
            'Art': 'count',
            'Sum_sum': 'sum',
            'Qty_sum': 'sum',
            'margin_mean': 'mean',
            'Price_mean': 'mean'
        }).rename(columns={'Art': 'sku_count'})

        stats['avg_gmv_per_sku'] = stats['Sum_sum'] / stats['sku_count']
        return stats

    def get_substitutes(self, art: str, top_n: int = 5) -> List[str]:
        """
        Найти SKU-заменители в том же сегменте.
        """
        sku_row = self.sku_df[self.sku_df['Art'] == art]
        if sku_row.empty:
            return []

        segment = sku_row.iloc[0]['Segment']
        price = sku_row.iloc[0]['Price_mean']

        # SKU в том же сегменте с похожей ценой (+/- 25%)
        candidates = self.sku_df[
            (self.sku_df['Segment'] == segment) &
            (self.sku_df['Art'] != art) &
            (self.sku_df['Price_mean'].between(price * 0.75, price * 1.25))
        ].copy()

        # Сортировка по GMV
        candidates = candidates.sort_values('Sum_sum', ascending=False)
        return candidates['Art'].head(top_n).tolist()

    def estimate_cannibalization_effect(self, removed_art: str) -> Dict[str, float]:
        """
        Оценка эффекта каннибализации при удалении SKU.
        """
        substitutes = self.get_substitutes(removed_art)
        if not substitutes:
            return {}

        # Распределение пропорционально GMV заменителей
        sub_df = self.sku_df[self.sku_df['Art'].isin(substitutes)]
        total_gmv = sub_df['Sum_sum'].sum()

        if total_gmv == 0:
            # Равномерное распределение
            return {art: 1.0 / len(substitutes) for art in substitutes}

        allocation = {}
        for _, row in sub_df.iterrows():
            allocation[row['Art']] = row['Sum_sum'] / total_gmv

        return allocation
