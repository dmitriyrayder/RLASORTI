"""
Feature Engineering для SKU оптимизации
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class SKUFeatureEngineering:
    """
    Создание признаков для RL агента.

    Признаки:
    - Оборачиваемость (turnover)
    - Маржинальность
    - Velocity (скорость продаж)
    - Сезонность
    - Позиция в сегменте
    - Stock coverage
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
        Создание признаков для SKU.

        Args:
            sku_df: DataFrame с агрегированными SKU (из data_loader.get_sku_aggregates)
            fit: Обучить ли scaler на этих данных

        Returns:
            DataFrame с добавленными признаками
        """
        df = sku_df.copy()

        # 1. Turnover (оборачиваемость)
        df['turnover'] = df['Qty_sum'] / df['num_transactions'].clip(lower=1)

        # 2. Velocity (скорость продаж)
        df['velocity'] = df['avg_daily_qty']

        # 3. Stock coverage (если есть данные о стоке)
        if 'Stock' in df.columns:
            df['stock_coverage_days'] = df['Stock'] / df['avg_daily_qty'].clip(lower=0.1)
        else:
            # Эвристика: 10% от недельных продаж
            df['estimated_stock'] = (df['avg_daily_qty'] * 7 * 0.1).clip(lower=1)
            df['stock_coverage_days'] = df['estimated_stock'] / df['avg_daily_qty'].clip(lower=0.1)

        # 4. Продуктовая давность (days since last sale)
        current_date = df['last_sale_date'].max()
        df['days_since_last_sale'] = (current_date - df['last_sale_date']).dt.days

        # 5. Contribution to segment GMV
        segment_gmv = df.groupby('Segment_<lambda>')['Sum_sum'].transform('sum')
        df['segment_contribution_pct'] = (df['Sum_sum'] / segment_gmv * 100).fillna(0)

        # 6. Price positioning в сегменте
        df['price_vs_segment_avg'] = df.groupby('Segment_<lambda>')['Price_mean'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

        # 7. Margin positioning
        df['margin_vs_segment_avg'] = df.groupby('Segment_<lambda>')['margin_mean'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

        # 8. ABC классификация по GMV
        df['abc_class'] = df.groupby('Segment_<lambda>')['Sum_sum'].transform(
            self._abc_classification
        )

        # 9. Частота продаж (transaction frequency)
        df['transaction_frequency'] = df['num_transactions'] / df['days_active']

        # 10. Average basket size
        df['avg_basket_size'] = df['avg_transaction_qty']

        # 11. Profit per day
        df['profit_per_day'] = df['total_profit'] / df['days_active']

        # 12. ROI
        df['roi'] = np.where(
            df['purchase_price_mean'] > 0,
            (df['margin_mean'] / df['purchase_price_mean']) * 100,
            0
        )

        # 13. Cannibalization risk (сколько других SKU в том же сегменте)
        df['segment_sku_count'] = df.groupby('Segment_<lambda>')['Art'].transform('count')
        df['cannibalization_risk'] = np.where(
            df['segment_sku_count'] > 10,
            np.log1p(df['segment_sku_count']) / 10,
            df['segment_sku_count'] / 10
        )

        # 14. Stability score (CV коэффициент вариации продаж)
        # Примерный расчет через соотношение средней к std
        df['stability_score'] = 1 / (1 + df['avg_transaction_qty'] / df['Qty_sum'].clip(lower=1))

        logger.info(f"Создано {len(df.columns) - len(sku_df.columns)} новых признаков")

        # Нормализация числовых признаков для RL
        numeric_features = [
            'turnover', 'velocity', 'stock_coverage_days', 'days_since_last_sale',
            'segment_contribution_pct', 'transaction_frequency', 'profit_per_day',
            'roi', 'cannibalization_risk', 'stability_score'
        ]

        if fit:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features].fillna(0))
            self.is_fitted = True
            self.feature_names = numeric_features
        elif self.is_fitted:
            df[numeric_features] = self.scaler.transform(df[numeric_features].fillna(0))

        return df

    @staticmethod
    def _abc_classification(series: pd.Series) -> pd.Series:
        """
        ABC классификация:
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
        Получить вектор состояния для конкретного SKU (для RL агента).

        Args:
            sku_row: Строка DataFrame с признаками SKU

        Returns:
            Numpy массив с нормализованными признаками
        """
        if not self.is_fitted:
            raise ValueError("Feature engineering не обучен. Используйте engineer_features с fit=True")

        features = [sku_row.get(f, 0) for f in self.feature_names]
        return np.array(features, dtype=np.float32)

    def get_state_dim(self) -> int:
        """Размерность вектора состояния"""
        return len(self.feature_names)


class SegmentAnalyzer:
    """
    Анализ сегментов для понимания эффектов каннибализации и замены.
    """

    def __init__(self, sku_df: pd.DataFrame):
        self.sku_df = sku_df
        self.segment_stats = self._compute_segment_stats()

    def _compute_segment_stats(self) -> pd.DataFrame:
        """Статистика по сегментам"""
        segment_col = 'Segment_<lambda>' if 'Segment_<lambda>' in self.sku_df.columns else 'Segment'

        stats = self.sku_df.groupby(segment_col).agg({
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

        Args:
            art: Артикул SKU
            top_n: Количество заменителей

        Returns:
            Список артикулов-заменителей
        """
        segment_col = 'Segment_<lambda>' if 'Segment_<lambda>' in self.sku_df.columns else 'Segment'

        sku_row = self.sku_df[self.sku_df['Art'] == art]
        if sku_row.empty:
            return []

        segment = sku_row.iloc[0][segment_col]
        price = sku_row.iloc[0]['Price_mean']

        # SKU в том же сегменте с похожей ценой (+/- 20%)
        candidates = self.sku_df[
            (self.sku_df[segment_col] == segment) &
            (self.sku_df['Art'] != art) &
            (self.sku_df['Price_mean'].between(price * 0.8, price * 1.2))
        ].copy()

        # Сортировка по GMV
        candidates = candidates.sort_values('Sum_sum', ascending=False)
        return candidates['Art'].head(top_n).tolist()

    def estimate_cannibalization_effect(self, removed_art: str) -> Dict[str, float]:
        """
        Оценка эффекта каннибализации при удалении SKU.

        Args:
            removed_art: Артикул удаляемого SKU

        Returns:
            Словарь {art: доля переключения спроса}
        """
        substitutes = self.get_substitutes(removed_art)
        if not substitutes:
            return {}

        segment_col = 'Segment_<lambda>' if 'Segment_<lambda>' in self.sku_df.columns else 'Segment'

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
