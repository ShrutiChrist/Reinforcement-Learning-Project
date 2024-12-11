# Multi-Armed Bandit Simulation
"Pulling the Right Strings: Multi-Armed Bandit Simulation"

## Overview
This project demonstrates the Multi-Armed Bandit problem, a fundamental concept in decision-making and reinforcement learning. The simulation implements four strategies to solve the problem:

- **Thompson Sampling**
- **Epsilon-Greedy**
- **Gradient Bandits**
- **Upper Confidence Bound (UCB)**

The interactive application, built using Streamlit, allows users to explore and compare the performance of these strategies based on metrics such as cumulative rewards and regret.

---

## Features
### Adaptable Simulation Settings
- **Number of Arms (k):** Adjustable between 2 and 20.
- **Iterations:** Configurable between 100 and 500 for analyzing short-term and long-term performance.
- **Strategies:** Choose from the four implemented decision-making strategies.
- **Epsilon (ε):** Configurable to control the exploration frequency in the Epsilon-Greedy approach.

---

## Strategies Implemented
1. **Upper Confidence Bound (UCB):**
   - Balances exploration and exploitation using statistical confidence intervals.
   - Prioritizes actions with high uncertainty and high estimated rewards.

2. **Gradient Bandits:**
   - Maximizes preferences for each arm using a gradient-based approach.
   - Updates probabilities based on rewards received.

3. **Epsilon-Greedy:**
   - Exploits the best-known action most of the time.
   - Occasionally explores other actions with a probability of ε.

4. **Thompson Sampling:**
   - Employs a Bayesian approach, sampling from a Beta distribution for each arm.
   - Selects actions based on probabilistic reward estimates.

---

## Installation and Setup

1. Install dependencies:
   ```bash
   pip install numpy
   pip install pandas
   pip install matplotlib
   pip install streamlit
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

---

## Usage
1. Launch the Streamlit app by running the command above.
2. Adjust the simulation settings, such as the number of arms, iterations, and strategy.
3. Observe the performance metrics, including cumulative rewards and regret, displayed through interactive visualizations.
4. Compare the effectiveness of different strategies.

---

## Key Metrics
- **Cumulative Rewards:** Total rewards accumulated over iterations.
- **Regret:** The difference between the maximum possible reward and the reward achieved by the chosen strategy.

---

## Applications
The Multi-Armed Bandit problem has real-world implications in:
- Clinical trials
- Resource allocation
- Internet advertising

This simulation provides insights into the trade-offs between exploration and exploitation, making it a valuable educational and practical tool.


---

## Contact
For questions or feedback, please contact:
- **Author:** Shruti
- **Linkedin URL:** [https://www.linkedin.com/in/shruti-ranjan20/]

