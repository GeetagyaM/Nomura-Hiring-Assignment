# imports
import pandas as pd
from tqdm import tqdm
import json
import numpy as np

# Assuming the availaible amount to be CNY 100M
AMOUNT = 100000000

######### Benchmark Performance #########
def calculate_benchmark_performance(start_time, end_time, stock_data, initial_amount):
        if start_time not in stock_data['time'].values:
            raise ValueError("Start time not found in the DataFrame.")
        if end_time not in stock_data['time'].values:
            raise ValueError("End time not found in the DataFrame.")

        n_stocks = stock_data['code'].unique().shape[0]
        per_stock_amount = initial_amount / n_stocks

        stock_prices_start = stock_data.loc[stock_data['time'] == start_time, ['code', 'close']].reset_index(drop=True)
        quantities_bought = per_stock_amount // stock_prices_start['close']
        money_invested_per_stock = stock_prices_start['close'] * quantities_bought
        money_invested = money_invested_per_stock.sum()
        money_left = initial_amount - money_invested

        stock_prices_end = stock_data.loc[stock_data['time'] == end_time, ['code', 'close']].reset_index(drop=True)
        quantities_sold = quantities_bought
        money_retrieved = (stock_prices_end['close'] * quantities_sold).sum()
        closing_amount = money_retrieved + money_left
        profit_pct_based_initial_amount = (closing_amount / initial_amount - 1) * 100

        result = {
            "n_stocks": n_stocks,
            "per_stock_amount": per_stock_amount,
            "money_left": money_left,
            "money_invested": money_invested,
            "money_retrieved": money_retrieved,
            "closing_amount": closing_amount,
            "profit_pct": profit_pct_based_initial_amount,
        }

        return result

######### Strategy Features #########
def calculate_sma(series , window):
    return series.rolling(window=window).mean()

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_zscore(series, window):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    z_score = (series - rolling_mean) / rolling_std
    return z_score

def calculate_rsi(series, window):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

########## Backtest #########
# Preparing the indicators based on hyperparamaeters
def prepare_indicators(df, long_window, long_to_short_ratio, long_ma_type, short_ma_type):

    short_window = long_window // long_to_short_ratio
    grouped = df.groupby('code')

    if long_ma_type == 'sma':
        long_ma_func = lambda x: calculate_sma(x, long_window)
    elif long_ma_type == 'ema':
        long_ma_func = lambda x: calculate_ema(x, long_window)
    else:
        raise ValueError("Invalid 'long_ma_type'. Use 'sma' or 'ema'.")

    if short_ma_type == 'sma':
        short_ma_func = lambda x: calculate_sma(x, short_window)
    elif short_ma_type == 'ema':
        short_ma_func = lambda x: calculate_ema(x, short_window)
    else:
        raise ValueError("Invalid 'short_ma_type'. Use 'sma' or 'ema'.")

    # Calculate selected moving averages, Z-Score, RSI for each group
    df['long MA'] = grouped['close'].apply(long_ma_func)
    df['short MA'] = grouped['close'].apply(short_ma_func)
    df['Z-Score'] = grouped['close'].apply(lambda x: calculate_zscore(x, long_window))
    df['RSI'] = grouped['close'].apply(lambda x: calculate_rsi(x, short_window))

    return df

# Preparing the filters based on the indicators prepared above
def prepare_filters(df, rsi_l, rsi_u, z_score_centre):
    df['MA_Down_Cross'] = (df['short MA'] < df['long MA']).astype(int)
    df['RSI_b/w_bounds'] = ((df['RSI'] >= rsi_l) & (df['RSI'] <= rsi_u)).astype(int)

    df['Z_Score_Trigger'] = 0
    z_score = df['Z-Score']

    # Apply the logic to set the trigger values
    df.loc[(z_score < 0) & (z_score >= -z_score_centre), 'Z_Score_Trigger'] = 1
    df.loc[(z_score < -z_score_centre) & (z_score >= -2*z_score_centre), 'Z_Score_Trigger'] = 2

    df['weights'] = df['MA_Down_Cross'] * df['RSI_b/w_bounds'] * df['Z_Score_Trigger']
    df['weights_normalized'] = df.groupby('time')['weights'].transform(lambda x: x / x.sum())
    df['weights_normalized'].fillna(0, inplace=True)


    return df

# Backtesting one particular config of hyper params
def trade(df, start_date, end_date, initial_amount):
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    time_stamps = df['time'].unique()

    portfolio = pd.DataFrame()
    portfolio['stocks'] = df['code'].unique()
    portfolio['quantities'] = 0

    trade_logs = pd.DataFrame(index = time_stamps)
    cash = initial_amount

    for ts in time_stamps:

        # Get the rows of the df for current timestamp
        current_timestamp = df[df['time']==ts]
        current_prices = current_timestamp['close'].reset_index(drop = True)

        # Find effective money availaible
        stock = (portfolio['quantities'] * current_prices).sum()
        effective_money_available = stock + cash

        # Get the ideal portfolio composition for current timestamp
        ideal_weights = current_timestamp['weights_normalized'].reset_index(drop = True)
        per_stock_amount = ideal_weights * effective_money_available
        ideal_composition = per_stock_amount//current_prices

        # Get the changes in composition for current timestamp
        changes_in_composition = ideal_composition - portfolio['quantities']

        # Update new quantities of the stocks in the portfolio and cash availaible
        portfolio['quantities'] = ideal_composition
        money_invested = (ideal_composition * current_prices).sum()
        cash = effective_money_available - money_invested

        # Update the trade logs
        trade_logs.loc[ts,'trxns']      = str(list(changes_in_composition))
        trade_logs.loc[ts,'stock']      = money_invested
        trade_logs.loc[ts,'cash']       = cash
        trade_logs.loc[ts,'eff_mon']    = effective_money_available

    return trade_logs

# Calling the three constituent functions(defined above) of a backtest
def run_trading_strategy(df, start_date, end_date, long_window, long_to_short_ratio, rsi_bounds, z_score_centre, long_ma_type, short_ma_type, initial_amount):

    df_with_indicators = prepare_indicators(df, long_window, long_to_short_ratio, long_ma_type, short_ma_type)
    df_with_filters = prepare_filters(df_with_indicators, rsi_bounds[0], rsi_bounds[1], z_score_centre)
    trade_logs = trade(df_with_filters, start_date, end_date, initial_amount)
    return trade_logs

######### Performance Metics Calculation #########
def calculate_metrics(value_series):

    total_return_percent = (value_series.iloc[-1]/value_series.iloc[0]-1) * 100

    cummax = value_series.cummax()
    drawdown = (value_series - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    return_series = value_series.pct_change()
    volatility = return_series.std() * np.sqrt(return_series.size) * 100

    result_dict = {
    'total_return_percent': total_return_percent,
    'volatility': volatility,
    'max_drawdown': max_drawdown,
    'return_per_unit_risk':total_return_percent/volatility
    }

    return result_dict

######### Main Function #########

if __name__ == '__main__':
    # Index composition of CSI500 index at date = '2021-01-01'
    file_path = './CSI Index Data/zz500_stocks_2021-01-01.csv'
    stock_composition = pd.read_csv(file_path)

    # 30min bar data from 2022-15-03 to 2022-07-31 for all 500 stocks of the CSI500 index
    # Taking a start date 15 days earlier to be able to calculate features to start trading on 2022-04-01
    file_path = "./CSI Index Data/stock_data.csv"
    stock_data_consolidated = pd.read_csv(file_path)

    start_date = '2022-03-15'
    end_date = '2022-07-31'

    # Find total trading days
    file_path = "./CSI Index Data/trading_days.csv"
    trading_days = pd.read_csv(file_path)
    print(f"\nTrading Days between {start_date} and {end_date}: ",trading_days['is_trading_day'].astype(int).sum())

    # Removing stocks for which 744 entries are not avaiable
    # 93(from 2022-15-03 to 2022-07-31) trading days and 8 timestamps per day
    code_counts = stock_data_consolidated['code'].value_counts()
    codes_with_less_than_744_entries = code_counts[code_counts < 744]

    stock_data_consolidated = stock_data_consolidated[~stock_data_consolidated['code'].isin(codes_with_less_than_744_entries.index)].reset_index(drop=True)
    print(f"\nShape of Stock Data after omitting stocks with missing entries: {stock_data_consolidated.shape}")

    # Calculate benchmark performance
    result = calculate_benchmark_performance(
        start_time=20220401100000000,
        end_time=20220630150000000,
        stock_data=stock_data_consolidated,
        initial_amount=AMOUNT
    )

    print(f"\nStarting Amount: {AMOUNT}")

    print("\nBenchmark Performance from 2022-04-01 to 2022-06-30:")
    for key,value in result.items():print(key,value)

    result_out_sample = calculate_benchmark_performance(
        start_time=20220701100000000,
        end_time=20220729150000000,
        stock_data=stock_data_consolidated,
        initial_amount=AMOUNT
    )
    print("\nBenchmark Performance from 2022-07-01 to 2022-07-29:")
    for key,value in result.items():print(key,value)

    # Backtest
    print("\nStarting Backtest")
    # Define the parameter combinations
    dates_list = [('2022-04-01', '2022-06-30'), ('2022-07-01', '2022-07-29')]
    long_ma_types = ['sma', 'ema']
    short_ma_types = ['sma', 'ema']
    long_windows = [60, 72, 84, 96]
    long_to_short_ratios = [2, 3, 4]
    rsi_bounds_list = [(20, 80), (30, 70), (40, 60)]
    z_score_centres = [1, 0.75]

    # Initialize the progress bar
    total_iterations = (
        len(dates_list)
        * len(long_ma_types)
        * len(short_ma_types)
        * len(long_windows)
        * len(long_to_short_ratios)
        * len(rsi_bounds_list)
        * len(z_score_centres)
    )
    pbar = tqdm(total=total_iterations)
    results = []
    index = 0
    # Iterate through parameter combinations
    for dates in dates_list:
        for long_ma_type in long_ma_types:
            for short_ma_type in short_ma_types:
                for long_window in long_windows:
                    for long_to_short_ratio in long_to_short_ratios:
                        for rsi_bounds in rsi_bounds_list:
                            for z_score_centre in z_score_centres:
                                trade_logs = run_trading_strategy(
                                    df=stock_data_consolidated,
                                    start_date=dates[0],
                                    end_date=dates[1],
                                    long_window=long_window,
                                    long_to_short_ratio=long_to_short_ratio,
                                    rsi_bounds=rsi_bounds,
                                    z_score_centre=z_score_centre,
                                    long_ma_type=long_ma_type,
                                    short_ma_type=short_ma_type,
                                    initial_amount=AMOUNT
                                )

                                # Save configuration and trade logs
                                config = {
                                    "dates": dates,
                                    "long_ma_type": long_ma_type,
                                    "short_ma_type": short_ma_type,
                                    "long_window": long_window,
                                    "long_to_short_ratio": long_to_short_ratio,
                                    "rsi_bounds": rsi_bounds,
                                    "z_score_centre": z_score_centre,
                                }
                                results.append({"config": config, "trade_logs": trade_logs})

                                with open(f"./JSONs/config_{index}.json", 'w') as config_file:
                                    json.dump(config, config_file)

                                # Save the trade logs as a CSV file
                                trade_logs.to_csv(f"./CSVs/trade_logs_{index}.csv", index=False)

                                # Update the progress bar
                                pbar.update(1)
                                index+=1

    # Close the progress bar
    pbar.close()
    print(f"\nSimulations Completed! Calculating Performance Metrics...")

    final_output = []
    for index in tqdm(range(total_iterations)):
        # Load the config from JSON
        with open(f"./JSONs/config_{index}.json", 'r') as config_file:
            config = json.load(config_file)

        # Read the trade logs from CSV
        trade_logs = pd.read_csv(f"./CSVs/trade_logs_{index}.csv")
        performance_metrics = calculate_metrics(trade_logs['eff_mon'])
        combined_data = {**config, **performance_metrics}
        final_output.append(combined_data)

    result_df = pd.DataFrame(final_output)
    result_df.to_csv(f"result_df_test.csv")





