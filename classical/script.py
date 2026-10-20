"""
Basic Monte Carlo pricing of a European call option.

Steps:
  1. Simulate many possible final stock prices S_T (geometric Brownian motion).
  2. Compute the payoff max(S_T - K, 0) for each one.
  3. Average the payoffs and discount back to today.
"""

import numpy as np

# Inputs
S0 = 100.0       # stock price today
K = 100.0        # strike price
r = 0.03         # risk-free interest rate
sigma = 0.2      # volatility
T = 1.0          # time to expiry (years)
N = 1_000_000    # number of simulations

# 1. Simulate final stock prices
Z = np.random.standard_normal(N)
S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

# 2. Payoff for each simulated price
payoff = np.maximum(S_T - K, 0)

# 3. Average and discount
price = np.exp(-r * T) * payoff.mean()
std_error = np.exp(-r * T) * payoff.std() / np.sqrt(N)

print(f"Option price: {price:.4f} +/- {std_error:.4f}")