import pandas as pd
import numpy as np
import yfinance as yf
import talib
from tqdm import tqdm
import os

direc = os.listdir('FutSF_TickKZ_CTP_Daily_202302')

for file_name in direc:
    df = pd.read_csv(os.path.join('FutSF_TickKZ_CTP_Daily_202302',file_name), encoding='gbk')
    print(file_name)
    f = df[['最新价','数量']].copy()
    f.columns = ['price','volume']
    f.loc[:,'time'] = pd.to_datetime(df['交易日'].astype(str) + ' ' + df['最后修改时间'])

    def tick_to_minute(df, freq="min"):
        """
        将Tick数据转换为分钟级OHLC数据
        :param tick_df: DataFrame, 需包含时间戳索引和price/volume列
        :param freq: 重采样频率（默认为1分钟）
        :return: 分钟级DataFrame
        """
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['time'])
        df.set_index('datetime', inplace=True)

        # 2. 使用 resample 将 tick 数据转换为分钟级数据
        # 计算价格的 OHLC 以及成交量的总和
        df_minute = df.resample(freq).agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })

        # 3. 整理列名称（可选）
        # df_minute.columns = ['_'.join(col).strip() for col in df_minute.columns.values]
        df_minute.columns = ['open', 'high', 'low', 'close', 'volume']
        df_minute = df_minute.reset_index()
        
        # 处理空值
        df_minute = df_minute.dropna()  # 或填充：minute_df.ffill()
        
        return df_minute

    def turtle_backtest(df, channel_period=20, atr_period=20, add_threshold=0.5, stop_loss_mult=2):
        """
        海龟交易法则回测示例（含加仓逻辑，不含尾盘平仓逻辑）
        
        参数说明：
        df: 包含 datetime, open, high, low, close, volume 的 DataFrame（索引为 datetime）
        channel_period: 唐奇安通道周期（例如20日）
        atr_period: ATR 计算周期（例如20日）
        add_threshold: 加仓阈值，单位为 ATR（例如0.5 表示价格比最近入场价上涨/下跌 0.5×ATR时加仓）
        stop_loss_mult: 止损倍数（例如2 表示价格反向移动 2×ATR时止损）
        
        返回：
        result: 增加了信号、持仓、累计盈亏等信息的 DataFrame
        trade_log: 交易记录列表，每笔记录包含入场日期、方向、入场价格列表、出场日期、出场价格及盈亏
        """
        df = df.copy()
        
        # 计算 ATR（周期为 atr_period）
        df['ATR'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=atr_period)
        
        # 计算唐奇安通道（不包含当前K线，所以使用 shift(1)）
        df['Donchian_High'] = pd.Series(talib.MAX(df['high'].values, timeperiod=channel_period)).shift(1)
        df['Donchian_Low']  = pd.Series(talib.MIN(df['low'].values, timeperiod=channel_period)).shift(1)
        
        # 初始化各列
        df['Signal'] = 0           # 信号：1 表示开多或加多，-1 表示开空或加空，0 表示平仓
        df['Position'] = 0         # 持仓数量（正数为多头，负数为空头）
        df['Trade_Price'] = np.nan # 当天发生交易时记录价格
        df['Cum_PnL'] = 0.0        # 累计盈亏
        
        position = 0             # 当前持仓数量
        entry_prices = []        # 记录每个入场单的价格（用于加仓及止损）
        last_entry = None        # 最近一次入场价格
        cumulative_pnl = 0.0
        trade_log = []           # 用于记录每笔交易详情
        
        # 从 channel_period 行开始遍历（前面数据不足无法计算通道）
        for i in range(channel_period, len(df)):
            current_date = df.index[i]
            close = df.iloc[i]['close']
            atr = df.iloc[i]['ATR']
            don_high = df.iloc[i]['Donchian_High']
            don_low = df.iloc[i]['Donchian_Low']
            
            # 如果关键指标未计算出来，则跳过
            if pd.isna(atr) or pd.isna(don_high) or pd.isna(don_low):
                df.at[current_date, 'Position'] = position
                df.at[current_date, 'Cum_PnL'] = cumulative_pnl
                continue
            
            # 无仓位时，判断入场信号
            if position == 0:
                if close > don_high:
                    # 突破上轨，开多仓
                    position = 1
                    entry_prices = [close]
                    last_entry = close
                    df.at[current_date, 'Signal'] = 1
                    df.at[current_date, 'Trade_Price'] = close
                    trade_log.append({
                        'Entry_Date': current_date,
                        'Direction': 'Long',
                        'Entry_Prices': entry_prices.copy()
                    })
                elif close < don_low:
                    # 突破下轨，开空仓
                    position = -1
                    entry_prices = [close]
                    last_entry = close
                    df.at[current_date, 'Signal'] = -1
                    df.at[current_date, 'Trade_Price'] = close
                    trade_log.append({
                        'Entry_Date': current_date,
                        'Direction': 'Short',
                        'Entry_Prices': entry_prices.copy()
                    })
            else:
                # 持仓中：判断加仓和止损
                if position > 0:  # 多仓逻辑
                    # 加仓条件：当前价格比最近入场价格上涨超过 add_threshold × ATR
                    if close >= last_entry + add_threshold * atr:
                        position += 1
                        entry_prices.append(close)
                        last_entry = close  # 更新最新入场价格
                        df.at[current_date, 'Signal'] = 1  # 表示加仓
                        df.at[current_date, 'Trade_Price'] = close
                        trade_log[-1]['Entry_Prices'] = entry_prices.copy()
                    # 止损条件：当前价格低于最近入场价格减去 stop_loss_mult × ATR
                    elif close <= last_entry - stop_loss_mult * atr:
                        pnl = sum([close - price for price in entry_prices])
                        cumulative_pnl += pnl
                        df.at[current_date, 'Signal'] = 0  # 平仓信号
                        df.at[current_date, 'Trade_Price'] = close
                        trade_log[-1].update({
                            'Exit_Date': current_date,
                            'Exit_Price': close,
                            'PnL': pnl
                        })
                        position = 0
                        entry_prices = []
                        last_entry = None
                elif position < 0:  # 空仓逻辑
                    # 加仓条件：当前价格比最近入场价格下跌超过 add_threshold × ATR
                    if close <= last_entry - add_threshold * atr:
                        position -= 1
                        entry_prices.append(close)
                        last_entry = close
                        df.at[current_date, 'Signal'] = -1  # 表示加仓
                        df.at[current_date, 'Trade_Price'] = close
                        trade_log[-1]['Entry_Prices'] = entry_prices.copy()
                    # 止损条件：当前价格高于最近入场价格加上 stop_loss_mult × ATR
                    elif close >= last_entry + stop_loss_mult * atr:
                        pnl = sum([price - close for price in entry_prices])
                        cumulative_pnl += pnl
                        df.at[current_date, 'Signal'] = 0  # 平仓信号
                        df.at[current_date, 'Trade_Price'] = close
                        trade_log[-1].update({
                            'Exit_Date': current_date,
                            'Exit_Price': close,
                            'PnL': pnl
                        })
                        position = 0
                        entry_prices = []
                        last_entry = None
                        
            # 记录当天的持仓和累计盈亏
            df.at[current_date, 'Position'] = position
            df.at[current_date, 'Cum_PnL'] = cumulative_pnl
        
        # 最后，在 for 循环结束后，若仍有未平仓仓位，则使用最后一根K线的价格平仓
        if position != 0:
            final_date = df.index[-1]
            final_price = df.iloc[-1]['close']
            if position > 0:
                pnl = sum([final_price - price for price in entry_prices])
            else:
                pnl = sum([price - final_price for price in entry_prices])
            cumulative_pnl += pnl
            # 更新最后一笔交易记录，标记为最终平仓
            if trade_log:
                trade_log[-1].update({
                    'Exit_Date': final_date,
                    'Exit_Price': final_price,
                    'PnL': pnl
                })
            df.at[final_date, 'Signal'] = 0
            df.at[final_date, 'Trade_Price'] = final_price
            df.at[final_date, 'Position'] = 0
            df.at[final_date, 'Cum_PnL'] = cumulative_pnl
            position = 0
            entry_prices = []
            last_entry = None
        
        return df, trade_log

    df = tick_to_minute(f, freq="min")

    total = []

    cur_max = 0
    # for cp in tqdm(range(2,40)):
    #     for ap in range(2,40):
    cp,ap = 4,4
    for at in [x / 10 for x in range(1,21)]:
        for sl in [x / 10 for x in range(2,24)]:
            # print(cp,ap,at,sl)
            result, trades = turtle_backtest(df, channel_period=cp, atr_period=ap, add_threshold=at, stop_loss_mult=sl)

            total_pnl = 0
            # 输出交易记录
            for trade in trades:
                total_pnl += trade.get('PnL', 0) - 0.6

            # print("总盈亏：", total_pnl)
            total.append([cp,ap,at,sl,total_pnl])
            if total_pnl > cur_max:
                cur_max = total_pnl
                print('\n', cp,ap,at,sl,total_pnl,len(trades))