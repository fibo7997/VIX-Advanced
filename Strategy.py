import nest_asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import *

# Apply nest_asyncio to allow async functions in Jupyter Notebooks
nest_asyncio.apply()

# Parameters
initial_capital = 10000
moving_window = 200  # Moving average window for Z-score in hours
max_position_pct = 0.3
z_score_entry_threshold = 1.5  # Tripled entry threshold for wider bands
z_score_exit_threshold = 0.6   # Tripled exit threshold for wider bands
take_profit_pct = 0.5  # 50% take-profit threshold
stop_loss_pct = 0.2  # 20% stop-loss threshold
min_return = 0.01  # Minimum return threshold for meta-labeling
num_threads = 1  # Number of threads for parallel processing
num_days = 3  # Maximum time window for vertical barrier (3 days)

# Load the Gamma data
gamma_data = pd.read_csv(r'C:\Users\Principal\Desktop\Tecoar\Synthetic_Gamma_Data.csv', parse_dates=['Date'], index_col='Date')
gamma_data = gamma_data.rename(columns={"Gamma (billions)": "daily_gamma"})

# Convert gamma_data index to timezone-naive to match VIX data
gamma_data.index = gamma_data.index.tz_localize(None)

# Reindex gamma data to hourly frequency over the full range of VIX data and fill missing values
vix_date_range = pd.date_range(gamma_data.index.min(), gamma_data.index.max(), freq='H')
gamma_data = gamma_data.reindex(vix_date_range).ffill().bfill()

# Connect to IBKR API
ib = IB()
try:
    ib.connect('xxx.0.0.1', 7496, clientId=xx)

    if ib.isConnected():
        print("Connected to IBKR API")

        # Define VIX contract
        contract = Index('VIX', 'CBOE')

        # Request hourly data for the last 3 years
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='3 Y',
            barSizeSetting='1 hour',
            whatToShow='TRADES',
            useRTH=True
        )

        # Convert data to DataFrame
        df = util.df(bars)
        df.set_index('date', inplace=True)

        # Ensure vix_data index is timezone-naive for merging
        df.index = df.index.tz_localize(None)

        # Calculate hourly returns
        vix_data = df[['close']].copy()
        vix_data['returns'] = vix_data['close'].pct_change().dropna()

        # Calculate moving Z-score of VIX price
        vix_data['moving_avg'] = vix_data['close'].rolling(moving_window).mean()
        vix_data['moving_std'] = vix_data['close'].rolling(moving_window).std()
        vix_data['moving_z_score'] = (vix_data['close'] - vix_data['moving_avg']) / vix_data['moving_std']
        vix_data.dropna(inplace=True)

        # Calculate daily volatility
        def getDailyVol(close, span0=100):
            df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
            df0 = df0[df0 > 0]
            valid_index = close.index[close.shape[0] - df0.shape[0]:]
            df0 = pd.Series(close.loc[valid_index].values / close.loc[close.index[df0]].values - 1, index=valid_index)
            df0 = df0.ewm(span=span0).std()
            return df0

        vix_data['daily_vol'] = getDailyVol(vix_data['close'])

        # Merge gamma data with VIX hourly data, filling any missing gamma values
        vix_data = vix_data.merge(gamma_data, how='left', left_index=True, right_index=True)
        vix_data['daily_gamma'].fillna(0, inplace=True)

        # Identify trade entry events based on Z-score threshold
        tEvents = vix_data.index[vix_data['moving_z_score'].abs() > z_score_entry_threshold]

        # Create a 3-day vertical barrier for each event
        def getVerticalBarrier(tEvents, close, numDays=3):
            verticalBarrier = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
            verticalBarrier = verticalBarrier[verticalBarrier < close.shape[0]]
            return pd.Series(close.index[verticalBarrier], index=tEvents[:verticalBarrier.shape[0]])

        # Calculate t1 as the vertical barrier for each trade entry event
        t1 = getVerticalBarrier(tEvents, vix_data['close'], numDays=num_days)
        print("Trade Entry and Vertical Barrier End Times (t1):\n", t1)

        # Define events with target (daily volatility) and minimum return threshold
        def getEvents(close, tEvents, ptSl, trgt, minRet, t1, side):
            trgt = trgt.loc[tEvents]
            trgt = trgt[trgt > minRet]
            t1 = pd.Series(t1, index=tEvents)
            events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side.loc[tEvents]}, axis=1).dropna(subset=['trgt'])
            return events

        # Define side labels based on Z-score
        side = pd.Series([-1 if vix_data['moving_z_score'].loc[event] > 0 else 1 for event in tEvents], index=tEvents)

        # Generate events for meta-labeling
        events = getEvents(
            close=vix_data['close'],
            tEvents=tEvents,
            ptSl=[take_profit_pct, stop_loss_pct],
            trgt=vix_data['daily_vol'],
            minRet=min_return,
            t1=t1,
            side=side
        )

        # Run the backtest with gamma-weighted positions
        def run_backtest():
            capital = initial_capital
            positions = 0
            entry_date = None
            entry_capital = 0
            entry_price = 0
            capital_history = []
            trade_log = []
            entry_points = []
            exit_points = []

            for i in range(len(vix_data)):
                date = vix_data.index[i]
                z_val = vix_data['moving_z_score'].iloc[i]
                close_price = vix_data['close'].iloc[i]
                gamma = vix_data['daily_gamma'].iloc[i]
                
                # Adjust max position based on gamma: more weight on short trades with high gamma and vice versa
                gamma_weight = max_position_pct * (1 + gamma) if gamma > 0 else max_position_pct * (1 - abs(gamma))
                max_capital_for_trade = gamma_weight * capital

                # Labeling: Determine the trade side
                side = None
                if z_val > z_score_entry_threshold and positions == 0:
                    # Short position label -1 with gamma-based weighting
                    side = -1
                    position_change = -max_capital_for_trade / close_price
                    entry_capital = max_capital_for_trade
                    entry_date = date
                    entry_price = close_price
                    capital += -position_change * vix_data['returns'].iloc[i] * close_price
                    positions = position_change
                    entry_points.append((date, entry_price))
                    trade_log.append({
                        'Entry Date': entry_date,
                        'Trade Type': 'Sell',
                        'Position Size': abs(position_change),
                        'Entry Price': entry_price,
                        'Moving Z-score': z_val,
                        'Side': side,
                        'Gamma': gamma  # Include gamma in the log
                    })

                elif z_val < -z_score_entry_threshold and positions == 0:
                    # Long position label 1 with gamma-based weighting
                    side = 1
                    position_change = max_capital_for_trade / close_price
                    entry_capital = max_capital_for_trade
                    entry_date = date
                    entry_price = close_price
                    capital += position_change * vix_data['returns'].iloc[i] * close_price
                    positions = position_change
                    entry_points.append((date, entry_price))
                    trade_log.append({
                        'Entry Date': entry_date,
                        'Trade Type': 'Buy',
                        'Position Size': abs(position_change),
                        'Entry Price': entry_price,
                        'Moving Z-score': z_val,
                        'Side': side,
                        'Gamma': gamma  # Include gamma in the log
                    })

                elif abs(z_val) <= z_score_exit_threshold and positions != 0:
                    # Exit based on Z-score returning to neutral range
                    exit_price = close_price
                    profit = positions * (exit_price - entry_price)
                    capital += profit
                    exit_points.append((date, exit_price))
                    trade_log[-1].update({
                        'Exit Date': date,
                        'Exit Price': exit_price,
                        'Profit': profit,
                        'Closing Reason': 'Z-score Exit',
                        'Moving Z-score': z_val
                    })
                    positions = 0

                     # Stop Loss and Take Profit
                if positions != 0:
                    current_price = close_price
                    trade_profit = positions * (current_price - entry_price)
                    profit_pct = trade_profit / entry_capital

                    # Check for take profit or stop loss
                    if profit_pct >= take_profit_pct:
                        capital += trade_profit
                        exit_points.append((date, current_price))
                        trade_log[-1].update({
                            'Exit Date': date,
                            'Exit Price': current_price,
                            'Profit': trade_profit,
                            'Closing Reason': 'Take Profit',
                            'Moving Z-score': z_val
                        })
                        positions = 0
                    elif profit_pct <= -stop_loss_pct:
                        capital += trade_profit
                        exit_points.append((date, current_price))
                        trade_log[-1].update({
                            'Exit Date': date,
                            'Exit Price': current_price,
                            'Profit': trade_profit,
                            'Closing Reason': 'Stop Loss',
                            'Moving Z-score': z_val
                        })
                        positions = 0

                capital_history.append(capital)

            # Create a DataFrame from trade_log
            trade_df = pd.DataFrame(trade_log)

            results = {
                'total_return': (capital - initial_capital) / initial_capital * 100,
                'trade_log': trade_df,
                'entry_points': entry_points,
                'exit_points': exit_points
            }

            return results

        # Function to analyze trades and save to Excel
        def analyze_trades(results, strategy_name="Gamma_Weighted_Strategy"):
            trade_log = results['trade_log']
            trade_log['Trade Length (hours)'] = (pd.to_datetime(trade_log['Exit Date']).dt.tz_localize(None) - 
                                                 pd.to_datetime(trade_log['Entry Date']).dt.tz_localize(None)).dt.total_seconds() / 3600
            trade_log['Profit (%)'] = (trade_log['Profit'] / (trade_log['Entry Price'] * trade_log['Position Size'])) * 100

            trade_log['Entry Date'] = pd.to_datetime(trade_log['Entry Date']).dt.tz_localize(None)
            trade_log['Exit Date'] = pd.to_datetime(trade_log['Exit Date']).dt.tz_localize(None)

            # Select and organize columns for better readability
            trade_log = trade_log[['Entry Date', 'Exit Date', 'Trade Type', 'Position Size', 
                                   'Entry Price', 'Exit Price', 'Profit', 'Profit (%)', 
                                   'Trade Length (hours)', 'Closing Reason', 'Side', 'Gamma']]
            
            output_filename = f"{strategy_name}_Trade_Stats4final.xlsx"
            trade_log.to_excel(output_filename, index=False)
            print(f"Trade statistics have been saved to {output_filename}")

        # Run the strategy and save its trade log to Excel
        strategy_results = run_backtest()
        analyze_trades(strategy_results, strategy_name="Gamma_Weighted_Strategy")

    else:
        print("Failed to connect to IBKR API.")

finally:
    ib.disconnect()
    print("Disconnected from IBKR API.")
