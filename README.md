# soc26-dynamic-pricing-rl
# Dynamic Pricing Using Reinforcement Learning

## Project Description
This project investigates dynamic pricing in competitive oligopoly markets using 
Reinforcement Learning and Multi-Agent Systems. The core objective is to study 
whether RL agents learn competitive (Nash) or collusive pricing strategies — 
replicating and extending the framework of Calvano et al. (2020).

## Objectives
- Build a custom Bertrand duopoly pricing environment using OpenAI Gymnasium
- Implement rule-based baseline agents (Random, Always-Nash, Always-Collude, Tit-for-Tat)
- Implement tabular Q-learning from scratch
- Study emergent collusion in multi-agent settings

## Tech Stack
- Python 3.10+
- NumPy, Pandas, Matplotlib
- Gymnasium (OpenAI)
- Reinforcement Learning (Tabular Q-Learning → DQN → PPO)

## Progress

### Week 1 — Game Theory Foundations
- Studied Nash Equilibrium, dominant strategies, Bertrand & Stackelberg competition
- Read Calvano et al. (2020) — Intro + Section 2
- Derived Bertrand Nash equilibrium analytically (price = marginal cost)
- Studied Prisoner's Dilemma and cooperation/defection dynamics

### Week 2 — Market Environment Design
- Designed custom Bertrand duopoly pricing environment
- Defined linear demand function: Q = a − bP
- Defined state space, action space (discrete price levels), and profit-based reward
- Built initial OpenAI Gymnasium-compatible environment class

### Week 3 — Rule-Based Baseline Agents
- Implemented Random Agent, Always-Nash Agent, Always-Collude Agent, Tit-for-Tat Agent
- Ran round-robin tournament across 1000+ rounds
- Generated result tables and performance plots

### Week 4 — Q-Learning Agent
- Implemented tabular Q-learning from scratch (no Stable-Baselines3)
- Coded Bellman update rule manually
- Used ε-greedy exploration with decay schedule (ε: 1.0 → 0.05)
- Q-learning agent consistently beats Random baseline ✅
- Mid-project review gate passed

## Future Work (Weeks 5–8)
- Deep Q-Networks (DQN)
- Proximal Policy Optimization (PPO)
- Self-play training
- Collusion detection and analysis

## Reference
Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020).
Artificial Intelligence, Algorithmic Pricing, and Collusion. *QJE*.
