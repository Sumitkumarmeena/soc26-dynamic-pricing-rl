"""
helper.py
Utility functions — SoC 2026 Dynamic Pricing RL
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# -----------------------------------------------------------------------
# Tournament runner
# -----------------------------------------------------------------------

def run_tournament(env, agent_a, agent_b, n_rounds: int = 1000, seed: int = 42):
    """
    Run a fixed-length tournament between two agents.

    Returns
    -------
    dict with keys: prices_a, prices_b, profits_a, profits_b, summary
    """
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)

    prices_a, prices_b = [], []
    profits_a, profits_b = [], []

    for _ in range(n_rounds):
        a0 = agent_a.act(obs)
        a1 = agent_b.act(obs)

        # TitForTat needs to observe opponent
        if hasattr(agent_a, "observe_opponent"):
            agent_a.observe_opponent(a1)
        if hasattr(agent_b, "observe_opponent"):
            agent_b.observe_opponent(a0)

        obs, rewards, terminated, truncated, info = env.step([a0, a1])

        prices_a.append(info["price_0"])
        prices_b.append(info["price_1"])
        profits_a.append(info["profit_0"])
        profits_b.append(info["profit_1"])

        if terminated or truncated:
            obs, _ = env.reset()

    summary = {
        "avg_price_a":  np.mean(prices_a),
        "avg_price_b":  np.mean(prices_b),
        "avg_profit_a": np.mean(profits_a),
        "avg_profit_b": np.mean(profits_b),
        "total_profit_a": np.sum(profits_a),
        "total_profit_b": np.sum(profits_b),
        "p_nash":    info["p_nash"],
        "p_collude": info["p_collude"],
    }

    return {
        "prices_a": prices_a, "prices_b": prices_b,
        "profits_a": profits_a, "profits_b": profits_b,
        "summary": summary,
    }


# -----------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------

def plot_price_history(result: dict, agent_a_name: str, agent_b_name: str,
                       env, save_path: str = None, last_n: int = 200):
    """Plot price trajectories of the last n rounds."""
    fig, ax = plt.subplots(figsize=(12, 4))
    n = min(last_n, len(result["prices_a"]))
    ax.plot(result["prices_a"][-n:], label=agent_a_name, alpha=0.8)
    ax.plot(result["prices_b"][-n:], label=agent_b_name, alpha=0.8)
    ax.axhline(env.p_nash,    color="red",   linestyle="--", label=f"Nash  P={env.p_nash:.2f}")
    ax.axhline(env.p_collude, color="green", linestyle="--", label=f"Collude P={env.p_collude:.2f}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Price")
    ax.set_title(f"Price Trajectories: {agent_a_name} vs {agent_b_name}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_training_curve(agent, window: int = 100, save_path: str = None):
    """Plot smoothed episode rewards during Q-learning training."""
    rewards = np.array(agent.episode_rewards)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(smoothed, color="steelblue")
    axes[0].set_title("Smoothed Episode Reward (Q-Learning)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel(f"Avg Reward (window={window})")

    axes[1].plot(agent.epsilon_history, color="darkorange")
    axes[1].set_title("Epsilon Decay")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("ε (Exploration Rate)")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_profit_comparison(results_dict: dict, env, save_path: str = None):
    """
    Bar chart comparing average profits across matchups.

    results_dict: { "MatchupLabel": tournament_result_dict, ... }
    """
    labels, profits_a, profits_b = [], [], []
    for label, res in results_dict.items():
        labels.append(label)
        profits_a.append(res["summary"]["avg_profit_a"])
        profits_b.append(res["summary"]["avg_profit_b"])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, profits_a, width, label="Agent A", color="steelblue")
    ax.bar(x + width/2, profits_b, width, label="Agent B", color="coral")

    # Nash and collude profit benchmarks
    nash_profit  = env._profit(env.p_nash, env.p_nash)
    col_profit   = env._profit(env.p_collude, env.p_collude)
    ax.axhline(nash_profit,  color="red",   linestyle="--", linewidth=1.2, label=f"Nash profit={nash_profit:.2f}")
    ax.axhline(col_profit,   color="green", linestyle="--", linewidth=1.2, label=f"Collude profit={col_profit:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Average Profit per Round")
    ax.set_title("Tournament Profit Comparison")
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


# -----------------------------------------------------------------------
# Collusion Index
# -----------------------------------------------------------------------

def collusion_index(avg_price: float, p_nash: float, p_collude: float) -> float:
    """
    Normalised collusion index ∈ [0, 1].
    0 = Nash (competitive), 1 = full collusion.
    """
    if abs(p_collude - p_nash) < 1e-8:
        return 0.0
    return np.clip((avg_price - p_nash) / (p_collude - p_nash), 0, 1)
