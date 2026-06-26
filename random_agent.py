"""
random_agent.py
Random Baseline Agent — SoC 2026 Dynamic Pricing RL
"""

import numpy as np


class RandomAgent:
    """
    Selects a price uniformly at random from the action space.
    Serves as the weakest possible baseline — any learning agent should beat this.
    Always seed before tournament runs for reproducibility.
    """

    def __init__(self, n_prices: int, seed: int = 0):
        self.n_prices = n_prices
        self.rng = np.random.default_rng(seed)
        self.name = "Random"

    def act(self, obs=None) -> int:
        return int(self.rng.integers(0, self.n_prices))

    def reset(self):
        pass  # stateless
