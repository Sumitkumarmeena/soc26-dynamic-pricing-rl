"""
q_learning_agent.py
Tabular Q-Learning Agent — SoC 2026 Dynamic Pricing RL

Implements Q-learning entirely from scratch (no Stable-Baselines3).
Based on: Watkins & Dayan (1992) and Sutton & Barto Ch. 6.5

Bellman Update Rule:
  Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

Key Design Choices (matching Calvano et al. 2020):
  - State: discretised (own_prev_price_bin, opp_prev_price_bin)
  - Action: discrete price index (0 … n_prices-1)
  - ε-greedy exploration: ε starts at 1.0, decays to 0.05 over 80% of training
  - Reward: normalised profit ∈ [0, 1]

Common Pitfalls (from Week 4 SOC notes):
  - Do NOT let epsilon decay too fast — premature exploitation kills convergence
  - Normalise rewards — raw profits cause Q-value explosion
  - Q-table can be large with fine price grid — consider state bucketing
"""

import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning agent for the Bertrand pricing environment.

    Parameters
    ----------
    n_prices      : number of discrete price actions
    n_state_bins  : number of bins to discretise each price dimension of state
    alpha         : learning rate (0 < α ≤ 1)
    gamma         : discount factor (0 ≤ γ < 1)
    epsilon_start : initial exploration rate
    epsilon_end   : final (minimum) exploration rate
    epsilon_decay : multiplicative decay per step
    seed          : random seed for reproducibility
    """

    def __init__(
        self,
        n_prices: int = 50,
        n_state_bins: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        seed: int = 42,
    ):
        self.n_prices = n_prices
        self.n_state_bins = n_state_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.name = "QLearning"

        # Q-table: shape = (state_bins, state_bins, n_prices)
        # state dimensions: (own_prev_price_bin, opp_prev_price_bin)
        self.q_table = np.zeros((n_state_bins, n_state_bins, n_prices))

        # Training history
        self.episode_rewards = []
        self.epsilon_history = []
        self.step_count = 0

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def _discretise_obs(self, obs) -> tuple:
        """
        Map continuous observation [own_norm, opp_norm] ∈ [0,1]²
        to discrete bin indices.
        """
        own_bin = int(np.clip(obs[0] * self.n_state_bins, 0, self.n_state_bins - 1))
        opp_bin = int(np.clip(obs[1] * self.n_state_bins, 0, self.n_state_bins - 1))
        return own_bin, opp_bin

    def act(self, obs) -> int:
        """ε-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_prices))   # explore
        state = self._discretise_obs(obs)
        return int(np.argmax(self.q_table[state]))             # exploit

    def update(self, obs, action: int, reward: float, next_obs, done: bool):
        """
        Bellman update:
          Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        s  = self._discretise_obs(obs)
        s_ = self._discretise_obs(next_obs)

        current_q  = self.q_table[s][action]
        target_q   = reward + (0 if done else self.gamma * np.max(self.q_table[s_]))
        td_error   = target_q - current_q

        self.q_table[s][action] += self.alpha * td_error

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.step_count += 1

    def reset(self):
        """Called at the start of each episode (not a full reset)."""
        pass

    def full_reset(self):
        """Resets Q-table and epsilon — use only when restarting training."""
        self.q_table = np.zeros((self.n_state_bins, self.n_state_bins, self.n_prices))
        self.epsilon = 1.0
        self.step_count = 0
        self.episode_rewards = []
        self.epsilon_history = []

    # ------------------------------------------------------------------
    # Training loop (standalone)
    # ------------------------------------------------------------------

    def train(self, env, n_episodes: int = 5000, agent_id: int = 0, opponent=None):
        """
        Train the Q-learning agent against a fixed opponent.

        Parameters
        ----------
        env        : BertrandPricingEnv instance
        n_episodes : number of training episodes
        agent_id   : 0 or 1 (which firm this agent controls)
        opponent   : another agent with .act(obs) method
                     If None, opponent plays randomly.
        """
        from src.agents.random_agent import RandomAgent
        if opponent is None:
            opponent = RandomAgent(env.n_prices, seed=0)

        opp_id = 1 - agent_id

        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                own_action = self.act(obs)
                opp_action = opponent.act(obs)

                actions = [own_action, opp_action] if agent_id == 0 else [opp_action, own_action]
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated

                reward = rewards[agent_id]
                self.update(obs, own_action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward

            self.episode_rewards.append(total_reward)
            self.epsilon_history.append(self.epsilon)

        print(f"Training complete. Final ε={self.epsilon:.4f}  |  "
              f"Avg reward (last 500 eps): "
              f"{np.mean(self.episode_rewards[-500:]):.4f}")

    def save_qtable(self, path: str = "results/logs/q_table.npy"):
        np.save(path, self.q_table)
        print(f"Q-table saved → {path}")

    def load_qtable(self, path: str = "results/logs/q_table.npy"):
        self.q_table = np.load(path)
        print(f"Q-table loaded ← {path}")
