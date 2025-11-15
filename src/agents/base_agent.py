"""
Базовый класс для RL агентов
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Абстрактный базовый класс для RL агентов.

    Все агенты должны реализовать методы:
    - select_action: выбор действия по состоянию
    - train_step: шаг обучения
    - save: сохранение модели
    - load: загрузка модели
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Args:
            state_dim: Размерность вектора состояния
            action_dim: Количество возможных действий
            learning_rate: Скорость обучения
            gamma: Коэффициент дисконтирования
            epsilon: Начальное значение epsilon для epsilon-greedy
            epsilon_min: Минимальное значение epsilon
            epsilon_decay: Коэффициент уменьшения epsilon
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.training_steps = 0
        self.episode_count = 0

        logger.info(
            f"Инициализирован {self.__class__.__name__}: "
            f"state_dim={state_dim}, action_dim={action_dim}"
        )

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Выбрать действие на основе текущего состояния.

        Args:
            state: Вектор состояния
            training: Флаг обучения (использовать exploration)

        Returns:
            Индекс выбранного действия
        """
        pass

    @abstractmethod
    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        Выполнить шаг обучения.

        Args:
            state: Текущее состояние
            action: Выполненное действие
            reward: Полученная награда
            next_state: Следующее состояние
            done: Флаг завершения эпизода

        Returns:
            Словарь с метриками обучения (loss и др.)
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Сохранить модель агента.

        Args:
            path: Путь для сохранения
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Загрузить модель агента.

        Args:
            path: Путь к сохраненной модели
        """
        pass

    def update_epsilon(self):
        """Уменьшить epsilon для epsilon-greedy стратегии"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def get_config(self) -> Dict:
        """Получить конфигурацию агента"""
        return {
            'agent_type': self.__class__.__name__,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'training_steps': self.training_steps,
            'episode_count': self.episode_count
        }


class ReplayBuffer:
    """
    Experience Replay Buffer для DQN и других off-policy алгоритмов.
    """

    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: Максимальный размер буфера
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Добавить опыт в буфер"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        """
        Сэмплировать батч из буфера.

        Args:
            batch_size: Размер батча

        Returns:
            (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self) -> int:
        return len(self.buffer)
