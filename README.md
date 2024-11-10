# VIX Trading Strategy with Gamma Overlay

This repository contains a quantitative trading strategy built around the VIX (Volatility Index) with a gamma overlay to improve trade precision. The strategy analyzes VIX price movements, using Z-score thresholds and gamma levels to determine entry and exit points.

Overview
This strategy leverages VIX price data and daily gamma exposure data to make trading decisions. Key features of the strategy include:

1. Gamma Overlay: Adjusts position sizes based on gamma levels. High gamma readings indicate a greater potential influence of market makers, so position sizes are increased when gamma is high.
   
2. Z-score Triggering: Uses moving Z-score of the VIX price to identify entry and exit points based on statistical deviations.

3. Risk Management: Implements take-profit and stop-loss levels to manage risk.
   

Strategy Details
  
1. VIX Data Processing:

Requests VIX price data and computes hourly returns.
Calculates a moving average and moving standard deviation of the VIX price, then derives a Z-score for triggering trades.

2. Gamma Data Integration:

Loads gamma data and interpolates it to match the frequency of VIX price data.
Uses the most recent available gamma values to fill gaps, ensuring that gamma values are always available for each VIX data point.
Adds a rate of change calculation for gamma to identify shifts in gamma levels.

3. Trade Entry and Exit Logic:

Entry: Trades are triggered when the VIX Z-score crosses the entry threshold. The strategy uses a short position if the Z-score is high and a long position if the Z-score is low.
Position Sizing: Adjusts position sizes based on the gamma level. Higher gamma levels result in larger position sizes to capitalize on expected volatility.
Exit: Exits are triggered when the Z-score reverts to within the exit threshold or if a take-profit or stop-loss condition is met.

4. Visualization:

The strategy plots VIX prices, gamma levels, gamma rate of change, and trade entry/exit points. This visualization aids in analyzing how gamma affects trade timing and the accuracy of entries/exits.

Files
strategy.py: Main script containing the trading strategy logic.
Synthetic_Gamma_Data.csv: Gamma data file, containing date and gamma values (in billions).
VIX_Gamma_Analysis.png: A saved visualization showing VIX prices, gamma levels, rate of change, and trade entry/exit points.
