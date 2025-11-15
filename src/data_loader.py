"""
Модуль загрузки и предобработки данных о продажах
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalesDataLoader:
    """
    Профессиональный загрузчик данных о продажах.

    Поддерживает:
    - Excel (.xlsx, .xls)
    - CSV
    - Валидацию схемы данных
    - Автоматическую очистку и типизацию
    """

    REQUIRED_COLUMNS = {
        'Magazin', 'Datasales', 'Art', 'Segment',
        'purchase_price', 'Price', 'Qty', 'Sum'
    }

    OPTIONAL_COLUMNS = {'Describe', 'Model', 'Stock', 'Category'}

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict = {}

    def load(self, file_path: str) -> pd.DataFrame:
        """
        Загрузка данных из файла.

        Args:
            file_path: Путь к файлу (Excel или CSV)

        Returns:
            DataFrame с очищенными данными
        """
        logger.info(f"Загрузка данных из {file_path}")

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path}")

        # Нормализация названий колонок
        df.columns = df.columns.str.strip()

        # Обратная совместимость с purchprice
        if 'purchprice' in df.columns and 'purchase_price' not in df.columns:
            df['purchase_price'] = df['purchprice']

        self._validate_schema(df)
        self.df = self._preprocess(df)
        self._compute_metadata()

        logger.info(f"Загружено {len(self.df)} записей, {self.df['Art'].nunique()} уникальных SKU")
        return self.df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Валидация схемы данных"""
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
        logger.info("Схема данных валидна")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных:
        - Типизация
        - Заполнение пропусков
        - Удаление аномалий
        - Вычисление производных полей
        """
        df = df.copy()

        # Дата продаж
        df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')
        df = df.dropna(subset=['Datasales'])

        # Числовые поля
        numeric_cols = ['Price', 'purchase_price', 'Qty', 'Sum']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Удаление отрицательных значений (возвраты отдельно)
        df = df[df['Qty'] >= 0]
        df = df[df['Price'] >= 0]

        # Пересчет Sum если не совпадает
        df['Sum_calculated'] = df['Price'] * df['Qty']
        df['Sum'] = df.apply(
            lambda x: x['Sum'] if abs(x['Sum'] - x['Sum_calculated']) < 0.01
            else x['Sum_calculated'],
            axis=1
        )
        df = df.drop('Sum_calculated', axis=1)

        # Маржа и маржинальность
        df['margin'] = df['Price'] - df['purchase_price']
        df['margin_pct'] = np.where(
            df['Price'] > 0,
            (df['margin'] / df['Price']) * 100,
            0
        )

        # Временные признаки
        df['year'] = df['Datasales'].dt.year
        df['month'] = df['Datasales'].dt.month
        df['quarter'] = df['Datasales'].dt.quarter
        df['week'] = df['Datasales'].dt.isocalendar().week
        df['dayofweek'] = df['Datasales'].dt.dayofweek

        # Артикул как строка
        df['Art'] = df['Art'].astype(str)

        logger.info("Предобработка завершена")
        return df

    def _compute_metadata(self) -> None:
        """Вычисление метаданных датасета"""
        self.metadata = {
            'total_records': len(self.df),
            'unique_skus': self.df['Art'].nunique(),
            'unique_stores': self.df['Magazin'].nunique(),
            'unique_segments': self.df['Segment'].nunique(),
            'date_range': (self.df['Datasales'].min(), self.df['Datasales'].max()),
            'total_gmv': self.df['Sum'].sum(),
            'total_qty': self.df['Qty'].sum(),
            'avg_price': self.df['Price'].mean(),
            'avg_margin_pct': self.df['margin_pct'].mean()
        }
        logger.info(f"Метаданные: {self.metadata}")

    def get_store_data(self, store_name: str) -> pd.DataFrame:
        """Получить данные по конкретному магазину"""
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте load() сначала.")
        return self.df[self.df['Magazin'] == store_name].copy()

    def get_sku_aggregates(self, store_name: Optional[str] = None) -> pd.DataFrame:
        """
        Агрегация по SKU для анализа.

        Args:
            store_name: Фильтр по магазину (опционально)

        Returns:
            DataFrame с агрегированными характеристиками SKU
        """
        df = self.get_store_data(store_name) if store_name else self.df

        agg_dict = {
            'Qty': 'sum',
            'Sum': 'sum',
            'Price': 'mean',
            'purchase_price': 'mean',
            'margin': 'mean',
            'margin_pct': 'mean',
            'Datasales': ['count', 'min', 'max'],
            'Magazin': 'first',
            'Segment': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
        }

        # Опциональные поля
        if 'Describe' in df.columns:
            agg_dict['Describe'] = 'first'
        if 'Model' in df.columns:
            agg_dict['Model'] = 'first'

        sku_agg = df.groupby('Art').agg(agg_dict)

        # Упрощение мультииндексов
        sku_agg.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col
            for col in sku_agg.columns.values
        ]

        # Переименование для читаемости
        rename_map = {
            'Datasales_count': 'num_transactions',
            'Datasales_min': 'first_sale_date',
            'Datasales_max': 'last_sale_date'
        }
        sku_agg = sku_agg.rename(columns=rename_map)

        # Дополнительные метрики
        days_active = (sku_agg['last_sale_date'] - sku_agg['first_sale_date']).dt.days + 1
        sku_agg['days_active'] = days_active.clip(lower=1)
        sku_agg['avg_daily_qty'] = sku_agg['Qty_sum'] / sku_agg['days_active']
        sku_agg['avg_transaction_qty'] = sku_agg['Qty_sum'] / sku_agg['num_transactions'].clip(lower=1)
        sku_agg['total_profit'] = sku_agg['margin_mean'] * sku_agg['Qty_sum']

        # Рейтинг по GMV в сегменте
        sku_agg['segment_rank'] = sku_agg.groupby('Segment_<lambda>')['Sum_sum'].rank(
            ascending=False, method='dense'
        )

        sku_agg = sku_agg.reset_index()

        logger.info(f"Агрегировано {len(sku_agg)} SKU")
        return sku_agg

    def get_summary_stats(self) -> Dict:
        """Получить сводную статистику"""
        if self.df is None:
            raise ValueError("Данные не загружены")

        return {
            **self.metadata,
            'top_skus_by_gmv': self.df.groupby('Art')['Sum'].sum().nlargest(10).to_dict(),
            'top_segments_by_gmv': self.df.groupby('Segment')['Sum'].sum().nlargest(10).to_dict(),
            'stores': self.df['Magazin'].unique().tolist()
        }
