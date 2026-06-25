"""
pricing_env.py
Custom Bertrand Duopoly Pricing Environment (OpenAI Gymnasium compatible)
SoC 2026 — Dynamic Pricing Using Reinforcement Learning

Based on: Calvano et al. (2020), QJE — "Artificial Intelligence, Algorithmic Pricing, and Collusion"
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BertrandPricingEnv(gym.Env):
    """
    Two-firm Bertrand duopoly pricing environment.

    Market Setup
    ------------
    - Two firms (Agent 0 and Agent 1) simultaneously set prices each period.
    - Demand: Q_i = a - b*P_i + c*P_j  (linear demand with cross-price effect)
    - Profit: π_i = (P_i - mc) * Q_i
    - Nash Equilibrium price: P_nash = (a + b*mc) / (2b - c)   [analytical]
    - Collusive price: P_collude = (a + b*mc) / (2*(b - c))    [joint monopoly]

    State Space
    -----------
    [P_agent0_prev, P_agent1_prev]  — both normalised to [0, 1]

    Action Space
    ------------
    Discrete: 0 … n_prices-1  →  mapped to [mc, P_max]

    Reward
    ------
    Normalised profit ∈ [0, 1] per agent per step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        a=10.0,          # demand intercept
        b=2.0,           # own-price sensitivity
        c=0.5,           # cross-price sensitivity (must satisfy 2b > c for stability)
        mc=1.0,          # marginal cost (both firms)
        p_max=8.0,       # maximum allowed price
        n_prices=50,     # number of discrete price levels per firm
        max_steps=200,   # episode length
        seed=42,
    ):
        super().__init__()

        # Market parameters
        self.a = a
        self.b = b
        self.c = c
        self.mc = mc
        self.p_max = p_max
        self.n_prices = n_prices
        self.max_steps = max_steps

        # Price grid: n_prices evenly spaced points in [mc, p_max]
        self.price_grid = np.linspace(mc, p_max, n_prices)

        # Analytical benchmarks
        self.p_nash = (a + b * mc) / (2 * b - c)
        self.p_collude = (a + b * mc) / (2 * (b - c))
        self.p_nash = np.clip(self.p_nash, mc, p_max)
        self.p_collude = np.clip(self.p_collude, mc, p_max)

        # Max possible profit (for normalisation)
        self._max_profit = self._profit(p_max, mc)

        # Spaces
        self.action_space = spaces.Discrete(n_prices)
        # Observation: [own_prev_price_norm, opp_prev_price_norm]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # Both firms start at Nash price
        nash_idx = np.argmin(np.abs(self.price_grid - self.p_nash))
        self.prev_actions = [nash_idx, nash_idx]
        obs = self._get_obs()
        return obs, {}

    def step(self, actions):
        """
        actions : list or tuple of two ints [action_agent0, action_agent1]
        Returns  : obs, [reward0, reward1], terminated, truncated, info
        """
        assert len(actions) == 2, "Provide actions for both agents."

        a0, a1 = int(actions[0]), int(actions[1])
        p0 = self.price_grid[a0]
        p1 = self.price_grid[a1]

        r0 = self._profit(p0, p1)
        r1 = self._profit(p1, p0)

        # Normalise rewards to [0, 1]
        r0_norm = np.clip(r0 / (self._max_profit + 1e-8), 0, 1)
        r1_norm = np.clip(r1 / (self._max_profit + 1e-8), 0, 1)

        self.prev_actions = [a0, a1]
        self.step_count += 1

        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {
            "price_0": p0, "price_1": p1,
            "profit_0": r0, "profit_1": r1,
            "p_nash": self.p_nash, "p_collude": self.p_collude,
        }
        return obs, [r0_norm, r1_norm], terminated, truncated, info

    def render(self, mode="human"):
        p0 = self.price_grid[self.prev_actions[0]]
        p1 = self.price_grid[self.prev_actions[1]]
        print(
            f"Step {self.step_count:4d} | "
            f"P0={p0:.2f}  P1={p1:.2f} | "
            f"Nash={self.p_nash:.2f}  Collude={self.p_collude:.2f}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _profit(self, p_self, p_other):
        """Profit for the firm pricing at p_self when opponent prices at p_other."""
        q = self.a - self.b * p_self + self.c * p_other
        q = max(q, 0.0)
        return (p_self - self.mc) * q

    def _get_obs(self):
        p0_norm = (self.price_grid[self.prev_actions[0]] - self.mc) / (self.p_max - self.mc)
        p1_norm = (self.price_grid[self.prev_actions[1]] - self.mc) / (self.p_max - self.mc)
        return np.array([p0_norm, p1_norm], dtype=np.float32)

    def get_nash_action(self):
        """Return the action index closest to the Nash equilibrium price."""
        return int(np.argmin(np.abs(self.price_grid - self.p_nash)))

    def get_collude_action(self):
        """Return the action index closest to the collusive price."""
        return int(np.argmin(np.abs(self.price_grid - self.p_collude)))
