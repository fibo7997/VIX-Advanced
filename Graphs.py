import nest_asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import *

# Apply nest_asyncio to allow async functions in Jupyter Notebooks
nest_asyncio.apply()

# Run the backtest and generate the strategy results (using your code structure)
strategy_results = run_backtest()

# Extract relevant data
vix_data = vix_data  # Already contains the merged data with daily gamma
entry_points = strategy_results['entry_points']
exit_points = strategy_results['exit_points']
trade_log = strategy_results['trade_log']

# Plot VIX, Gamma, and Trade Entry/Exit Points
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot VIX price on primary y-axis
ax1.plot(vix_data.index, vix_data['close'], label='VIX Close Price', color='blue', linewidth=1)
ax1.set_xlabel('Date')
ax1.set_ylabel('VIX Close Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot trade entry and exit points on VIX
for entry in entry_points:
    ax1.plot(entry[0], entry[1], marker='^', color='green', markersize=8, label='Entry Point' if entry == entry_points[0] else "")
for exit in exit_points:
    ax1.plot(exit[0], exit[1], marker='v', color='red', markersize=8, label='Exit Point' if exit == exit_points[0] else "")

# Plot gamma on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(vix_data.index, vix_data['daily_gamma'], label='Daily Gamma', color='orange', linestyle='--', linewidth=1)
ax2.set_ylabel('Gamma (billions)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add legends
fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85), bbox_transform=ax1.transAxes)

# Add title and grid
plt.title('VIX with Gamma and Trade Entry/Exit Points')
plt.grid(True)

plt.show()

# Analyze a few example trades with annotations
# Filter trades with high gamma for a trade explanation
example_trades = trade_log[trade_log['Gamma'] > 1].head(3)  # Example: select trades where gamma > 1

# Show trade logic
for idx, trade in example_trades.iterrows():
    print(f"Trade on {trade['Entry Date']}")
    print(f" - Type: {'Buy' if trade['Side'] == 1 else 'Sell'}")
    print(f" - Entry Price: {trade['Entry Price']}")
    print(f" - Exit Price: {trade['Exit Price']}")
    print(f" - Profit: {trade['Profit']}")
    print(f" - Position Size: {trade['Position Size']}")
    print(f" - Gamma: {trade['Gamma']}")
    if trade['Gamma'] > 0.5:
        print(" - We saw high gamma, so we increased the position size to capture more potential profit.\n")
    else:
        print(" - Gamma was low, so the position size was kept moderate to reduce risk.\n")
