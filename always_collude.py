"""
always_collude.py
Always-Collude Baseline Agent — SoC 2026 Dynamic Pricing RL

Sets price equal to the joint monopoly (collusive) price every single period.
Represents the best-case outcome IF both firms cooperate.
Collusive price = (a + b*mc) / (2*(b - c))

Warning: Always-Collude "wins" in short runs — run 1000+ rounds so defection
incentives emerge properly (Pitfall from Week 3 SOC notes).
"""


class AlwaysColludeAgent:
    """
    Always plays the collusive (joint monopoly) price.
    Highest possible profit when BOTH firms collude.
    Unstable against defectors — Folk Theorem explains why.
    """

    def __init__(self, env):
        self.collude_action = env.get_collude_action()
        self.name = "AlwaysCollude"

    def act(self, obs=None) -> int:
        return self.collude_action

    def reset(self):
        pass  # stateless
