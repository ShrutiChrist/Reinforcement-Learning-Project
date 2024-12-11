import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


st.title("Multi-Armed Bandit Simulation with Multiple Strategies")

# User Inputs
k = st.sidebar.slider("Number of Arms", 2, 20, 10)
iterations = st.sidebar.slider("Number of Iterations", 100, 300, 500)
strategy = st.sidebar.selectbox(
    "Strategy", 
    ["Upper Confidence Bound (UCB)", "Gradiecmnt Bandits", "Epsilon-Greedy", "Thompson Sampling"]
)
epsilon = st.sidebar.slider("Epsilon (for Epsilon-Greedy)", 0.01, 1.0, 0.1)

# Simulate the Multi-Armed Banditc
true_rewards = np.random.rand(k)  # True reward probabilities for each arm
estimated_rewards = np.zeros(k)  # Estimated rewards for each arm
counts = np.zeros(k)  # Number of times each arm is pulled
alpha = np.ones(k)  # For Thompson Sampling (Beta distribution alpha)
beta = np.ones(k)   # For Thompson Sampling (Beta distribution beta)
total_reward = 0
regret = []

for t in range(1, iterations + 1):
    if strategy == "Upper Confidence Bound (UCB)":
        ucb_values = estimated_rewards + np.sqrt(2 * np.log(t) / (counts + 1e-5))
        action = np.argmax(ucb_values)

    elif strategy == "Gradient Bandits":
        preferences = np.exp(estimated_rewards)
        probabilities = preferences / np.sum(preferences)
        action = np.random.choice(np.arange(k), p=probabilities)

    elif strategy == "Epsilon-Greedy":
        if np.random.rand() < epsilon:
            action = np.random.randint(0, k)  # Explore
        else:
            action = np.argmax(estimated_rewards)  # Exploit

    elif strategy == "Thompson Sampling":
        samples = np.random.beta(alpha, beta)
        action = np.argmax(samples)

    # Simulate reward
    reward = np.random.normal(true_rewards[action], 0.1)

    # Update statistics
    counts[action] += 1
    total_reward += reward
    regret.append(np.max(true_rewards) * t - total_reward)

    # Update strategy-specific parameters
    if strategy in ["Upper Confidence Bound (UCB)", "Epsilon-Greedy", "Gradient Bandits"]:
        estimated_rewards[action] += (reward - estimated_rewards[action]) / counts[action]
    elif strategy == "Thompson Sampling":
        if reward > 0.5:
            alpha[action] += 1
        else:
            beta[action] += 1

# Visualization
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot True vs Estimated Rewards
ax[0].bar(range(k), true_rewards, alpha=0.6, label="True Rewards")
ax[0].bar(range(k), estimated_rewards, alpha=0.6, label="Estimated Rewards")
ax[0].set_title("True vs Estimated Rewards")
ax[0].legend()

# Plot Regret Over Time
ax[1].plot(regret, label="Cumulative Regret")
ax[1].set_title("Regret Over Time")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Regret")
ax[1].legend()

st.pyplot(fig)
