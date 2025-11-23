"""
DQN Agent для оптимизации ассортимента
Объединенная версия без отдельной структуры папок
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience Replay Buffer"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Добавить опыт в буфер"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Сэмплировать батч из буфера"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-Network (нейронная сеть для аппроксимации Q-функции)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple = (128, 64)):
        super(QNetwork, self).__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network агент с Experience Replay и Target Network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden_dims: tuple = (128, 64),
        use_double_dqn: bool = True,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.device = torch.device(device)
        self.training_steps = 0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Q-Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        logger.info(
            f"DQN инициализирован: buffer_size={buffer_size}, "
            f"batch_size={batch_size}, double_dqn={use_double_dqn}"
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Выбрать действие с помощью epsilon-greedy стратегии"""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """Шаг обучения DQN"""
        # Сохранить опыт в replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Обучение только если достаточно данных
        if len(self.replay_buffer) < self.batch_size:
            return {'loss': 0.0, 'q_mean': 0.0}

        # Сэмплировать батч
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Конвертация в тензоры
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Текущие Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Целевые Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_network(next_states).max(dim=1)[0]

            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss (Huber loss для стабильности)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Обновление счетчика
        self.training_steps += 1

        # Обновление target network
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            'loss': loss.item(),
            'q_mean': current_q_values.mean().item(),
            'q_max': current_q_values.max().item()
        }

    def update_epsilon(self):
        """Обновить epsilon для exploration"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """Сохранить модель агента"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }
        torch.save(checkpoint, f"{path}.pth")
        logger.info(f"Модель сохранена в {path}.pth")

    def load(self, path: str):
        """Загрузить модель агента"""
        checkpoint = torch.load(f"{path}.pth", map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        logger.info(f"Модель загружена из {path}.pth")

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Получить Q-values для всех действий"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]
