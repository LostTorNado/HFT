import pandas as pd
import numpy as np
import talib
import itertools
from multiprocess import Pool
import os

class Variable:

    def __init__(self,para,df,p = 'LastPrice'):
        self.para = para
        self.df = df
        self.SRevert = -np.inf
        self.BRevert = np.inf
        self.div = 3
        self.ifSetup = None
        self.p = p
    
    def CalFixedBands(self):
        r1,r2,r3 = self.para
        H,L,C = max(self.df[self.p]),min(self.df[self.p]),self.df[self.p].iloc[-1]

        self.SEnter = ((1 + r1)/2 *(H + C)) - r1 * L
        self.BEnter = ((1 + r1) / 2 * (L + C)) - r1 * H
        self.SSetup = H + r2 * (C - L)
        self.BSetup = L - r2 * (H - C)
        self.BBreak = self.SSetup + r3 * (self.SSetup - self.BSetup)
        self.SBreak = self.BSetup - r3 * (self.SSetup - self.BSetup)
    
    def CalFloatingBand(self,tdhigh,tdlow):
        self.SRevert = self.SEnter + (tdhigh - self.SSetup) / self.div
        self.BRevert = self.BEnter + (tdlow - self.BSetup) / self.div

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
    df_minute.columns = ['open', 'high', 'low', 'close', 'volume']
    df_minute = df_minute.reset_index()
    
    # 处理空值
    df_minute = df_minute.dropna()  # 或填充：minute_df.ffill()
    
    return df_minute

def rbreaker_backtest(df,pref, para = [0.01,0.01,0.01], atr_period=2, add_threshold=0.5, stop_loss_mult=2, cost = 0.6):

    if df['InstrumentID'][0] != pref['InstrumentID'][0]:
        return
    
    df,pref = df.copy(),pref.copy()

    variable = Variable(para,pref)
    variable.CalFixedBands()
    
    f = df[['LastPrice','Volume']].copy()
    f.columns = ['price','volume']
    f.loc[:,'time'] = pd.to_datetime(df['TradingDay'].astype(str) + ' ' + df['UpdateTime'])
    df = tick_to_minute(f, freq="min").iloc[1:-2,:]

    # 计算 ATR（周期为 atr_period）
    df['ATR'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=atr_period)
    
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
    for i in range(len(df)):
        current_date = df.index[i]
        close,high,low = df.iloc[i]['close'],df.iloc[i]['high'],df.iloc[i]['low']
        atr = df.iloc[i]['ATR']
        
        if high > variable.BBreak and variable.BBreak > variable.SRevert:
            if position < 0:
                pnl = sum([price - close - cost for price in entry_prices])
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

            if position == 0:
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

        if low < variable.SBreak and variable.SBreak < variable.BRevert:
            if position > 0:
                pnl = sum([close - price - cost for price in entry_prices])
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

            if position == 0:
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

        if variable.ifSetup == 'BSetup':
            if high > variable.BRevert:
                if position < 0:
                    pnl = sum([price - close - cost for price in entry_prices])
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

                if position == 0:
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
        if variable.ifSetup == 'SSetup':
            if low < variable.SRevert:
                if position > 0:
                    pnl = sum([close - price - cost for price in entry_prices])
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

                    if position == 0:
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
        if position > 0:
            if high >= last_entry + add_threshold * atr:
                position += 1
                entry_prices.append(close)
                last_entry = close  # 更新最新入场价格
                df.at[current_date, 'Signal'] = 1  # 表示加仓
                df.at[current_date, 'Trade_Price'] = close
                trade_log[-1]['Entry_Prices'] = entry_prices.copy()
            # 止损条件：当前价格低于最近入场价格减去 stop_loss_mult × ATR
            elif low <= last_entry - stop_loss_mult * atr:
                pnl = sum([close - price - cost for price in entry_prices])
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
        
        if position < 0:
            if low <= last_entry - add_threshold * atr:
                position -= 1
                entry_prices.append(close)
                last_entry = close
                df.at[current_date, 'Signal'] = -1  # 表示加仓
                df.at[current_date, 'Trade_Price'] = close
                trade_log[-1]['Entry_Prices'] = entry_prices.copy()

            elif high >= last_entry + stop_loss_mult * atr:
                pnl = sum([price - close - cost for price in entry_prices])
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

        variable.CalFloatingBand(df['high'].iloc[i],df['low'].iloc[i])
        if high > variable.SSetup:
            variable.ifSetup = 'SSetup'
        if low < variable.BSetup:
            variable.ifSetup = 'BSetup'

        # 记录当天的持仓和累计盈亏
        df.at[current_date, 'Position'] = position
        df.at[current_date, 'Cum_PnL'] = cumulative_pnl
    
    # 最后，在 for 循环结束后，若仍有未平仓仓位，则使用最后一根K线的价格平仓
    if position != 0:
        final_date = df.index[-1]
        final_price = df.iloc[-1]['close']
        if position > 0:
            pnl = sum([final_price - price - cost for price in entry_prices])
        else:
            pnl = sum([price - final_price - cost for price in entry_prices])
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

wholef = pd.DataFrame()
direc = os.listdir('FutSF_TickKZ_CTP_Daily_202302')

for file_name in direc:
    if 'IC' not in file_name:
        continue
    df = pd.read_csv(os.path.join('FutSF_TickKZ_CTP_Daily_202302',file_name), encoding='gbk')

    f = df[['最新价','数量']].copy()
    f.columns = ['price','volume']
    f.loc[:,'time'] = pd.to_datetime(df['交易日'].astype(str) + ' ' + df['最后修改时间'])
    ff = tick_to_minute(f, freq="min").iloc[:-4,:]
    wholef=pd.concat([wholef,ff],ignore_index=True)

def func(ca,asltotal,total = []):
    cp,ap = ca
    cur_max = 0
    # for at in [x / 10 for x in range(1,30)]:
    #     for sl in [x / 10 for x in range(2,30)]:
    def subfunc(atsl):
        at,sl = atsl
        result, trades = rbreaker_backtest(wholef, channel_period=cp, atr_period=ap, add_threshold=at, stop_loss_mult=sl)

        total_pnl = 0
        # 输出交易记录
        for trade in trades:
            total_pnl += trade.get('PnL', 0) - 0.6

        # print("总盈亏：", total_pnl)
        # total.append([cp,ap,at,sl,total_pnl])
        # if total_pnl > cur_max:
        #     cur_max = total_pnl
        # if total_pnl > 0:
        #     print('\n', cp,ap,at,sl,total_pnl,f'cur_max: {cur_max}',len(trades))
        return [cp,ap,at,sl,total_pnl,len(trades)]
    with Pool(processes = 4) as pool:
        result = pool.map(lambda atsl:subfunc(atsl), atsltotal)
    result = sorted(result,key=lambda x:x[4],reverse=True)
    for r in result[:15]:
        if r[4] > cur_max:
            cur_max = r[4]
        if r[4] > 0:
            print('\n', r[0],r[1],r[2],r[3],r[4],f'cur_max: {cur_max}',r[5])
    total = np.append(total,result[:10])
    np.save('total.npy', total)
    return total
# np.save(total, 'total.npy')

if __name__ == '__main__':

    cpap = list(itertools.product(range(2,40),range(2,40)))
    atsltotal = list(itertools.product([x / 10 for x in range(1,30)],[x / 10 for x in range(2,30)]))
    # with Pool(processes = 10) as pool:
    #     resultss = pool.map(lambda cp:func(cp), cpap)
    resultss = []
    r = []
    for c in cpap:
        r = func(c,atsltotal,r)
    resultss = np.append(resultss,r)
    np.save(resultss, 'resultss.npy')