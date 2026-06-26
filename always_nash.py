"""
always_nash.py
Always-Nash Baseline Agent — SoC 2026 Dynamic Pricing RL

Sets price equal to the Bertrand Nash Equilibrium price every single period.
Represents rational competitive play under perfect information.
Nash price = (a + b*mc) / (2b - c)
"""


class AlwaysNashAgent:
    """
    Always plays the Nash Equilibrium price action.
    Used as the competitive benchmark — prices are driven to marginal cost level.
    """

    def __init__(self, env):
        self.nash_action = env.get_nash_action()
        self.name = "AlwaysNash"

    def act(self, obs=None) -> int:
        return self.nash_action

    def reset(self):
        pass  # stateless
