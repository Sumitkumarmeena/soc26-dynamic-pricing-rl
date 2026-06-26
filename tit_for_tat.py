"""
tit_for_tat.py
Tit-for-Tat Agent — SoC 2026 Dynamic Pricing RL

Based on Axelrod (1980, 1984) — winner of the original Prisoner's Dilemma tournament.
Four key properties (Axelrod):
  1. Nice      — starts by cooperating (collusive price)
  2. Retaliating — immediately matches defection (cuts price to Nash)
  3. Forgiving  — returns to collude once opponent returns to high price
  4. Clear      — simple and predictable strategy

In pricing context:
  - "Cooperate"  = price at or above collude threshold → reward cooperation
  - "Defect"     = price below collude threshold       → punish with Nash price
"""


class TitForTatAgent:
    """
    Tit-for-Tat pricing strategy.
    Starts at collusive price. Mirrors opponent's last-period behaviour.
    """

    def __init__(self, env, collude_threshold: float = 0.8):
        """
        Parameters
        ----------
        env              : BertrandPricingEnv instance
        collude_threshold: fraction of collude price below which opponent is deemed a defector
        """
        self.env = env
        self.nash_action = env.get_nash_action()
        self.collude_action = env.get_collude_action()
        self.collude_threshold = collude_threshold

        # Defect price threshold (action index)
        self.defect_action_threshold = int(
            self.collude_action * collude_threshold
        )
        self.last_opponent_action = self.collude_action  # start cooperating
        self.name = "TitForTat"

    def act(self, obs=None) -> int:
        """
        If opponent cooperated last round → collude.
        If opponent defected last round   → Nash (punish).
        """
        if self.last_opponent_action >= self.defect_action_threshold:
            return self.collude_action   # cooperate
        else:
            return self.nash_action      # retaliate

    def observe_opponent(self, opponent_action: int):
        """Call this after each step to update opponent's last action."""
        self.last_opponent_action = opponent_action

    def reset(self):
        self.last_opponent_action = self.collude_action
