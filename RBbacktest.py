import pandas as pd
import numpy as np
import talib
import itertools
from multiprocess import Pool
import os
from tqdm import tqdm
import gc

class Variable:

    def __init__(self,para,df):
        self.para = para
        self.df = df
        self.SRevert = -np.inf
        self.BRevert = np.inf
        self.div = 3
        self.ifSetup = None
    
    def CalFixedBands(self):
        r1,r2,r3 = self.para
        H,L,C = max(self.df['LastPrice']),min(self.df['LastPrice']),self.df['LastPrice'].iloc[-1]

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

def rbreaker_backtest(df,pref,init_pos = 0, para = [0.01,0.01,0.01], atr_period=2, add_threshold=0.5, stop_loss_mult=2, cost = 0.6):
    
    if df['InstrumentID'][0] != pref['InstrumentID'][0]:
        return None,None
    
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

        if init_pos > 0:
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
            continue
        
        if init_pos < 0:
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
            continue
        
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

# for file_name in direc:
#     if 'IM' not in file_name:
#         continue
#     df = pd.read_csv(os.path.join(fpath,file_name), encoding='gbk')

#     f = df[['最新价','数量']].copy()
#     f.columns = ['price','volume']
#     f.loc[:,'time'] = pd.to_datetime(df['交易日'].astype(str) + ' ' + df['最后修改时间'])
#     ff = tick_to_minute(f, freq="min").iloc[:-4,:]
#     wholef=pd.concat([wholef,ff],ignore_index=True)

def func(ca,rtotal,total = [],code = 'IM'):
    if not os.path.exists(f'results/{code}'):
        os.makedirs(f'results/{code}',exist_ok=True)
    ap,at,sl = ca
    cur_max = 0
    fpath = 'data/' + code
    direc = os.listdir(fpath)
    reftd = pd.read_csv(f'preds/{code}.csv')
    startdate = min([int(i.split('.')[0].split('_')[1]) for i in direc])
    enddate = max([int(i.split('.')[0].split('_')[1]) for i in direc])
    reftd = reftd[(pd.to_datetime(reftd['Date']) >= pd.to_datetime(str(startdate))) & 
                  (pd.to_datetime(reftd['Date']) <= pd.to_datetime(str(enddate)))].reset_index(drop=True)

    for i in tqdm(range(len(reftd) - 1)):
        if os.path.exists(f'results/{code}/{reftd['Date'][i+1]}.csv'):
            continue
        prename = f'{code}_{pd.to_datetime(reftd["Date"][i]).strftime("%Y%m%d")}.csv'
        name = f'{code}_{pd.to_datetime(reftd["Date"][i+1]).strftime("%Y%m%d")}.csv'
        if prename not in direc or name not in direc:
            continue
        pref = pd.read_csv(os.path.join(fpath,prename), encoding='gbk')
        df = pd.read_csv(os.path.join(fpath,name), encoding='gbk')
        pred = reftd['pred'][i]

        def subfunc(r1r2r3):

            result, trades = rbreaker_backtest(df, pref, pred, r1r2r3, atr_period=ap, add_threshold=at, stop_loss_mult=sl)

            day_pnl = 0
            day_trades = 0
            if trades is not None:
                for trade in trades:
                    day_pnl += trade.get('PnL', 0)
                day_trades += len(trades)
            else:
                day_pnl = -1000000
                day_trades = -1

            return [r1r2r3[0],r1r2r3[1],r1r2r3[2],ap,at,sl,day_pnl,day_trades]
        gc.collect()
        with Pool(processes = 16) as pool:
            result = pool.map(lambda atsl:subfunc(atsl), rtotal)
        # result = []
        # for r in tqdm(rtotal):
        #     result.append(subfunc(r))
        
        to_save = result
        to_save = pd.DataFrame(to_save,columns=['r1','r2','r3','ap','at','sl','pnl','trades'])
        to_save = to_save.sort_values(by = ['pnl'],ascending=False)

        if not os.path.exists(f'results/{code}/{reftd['Date'][i+1]}.csv'):
            to_save.to_csv(f'results/{code}/{reftd['Date'][i+1]}.csv',index=False)
        else:
            pre = pd.read_csv(f'results/{code}/{reftd['Date'][i+1]}.csv')
            pre = pd.concat([pre,to_save],ignore_index=True)
            pre = pre.sort_values(by = ['pnl'],ascending=False)
            pre.to_csv(f'results/{code}/{reftd['Date'][i+1]}.csv',index=False)

    result = sorted(result,key=lambda x:x[6],reverse=True)
    # csvfile = pd.DataFrame(result,columns=['r1','r2','r3','ap','at','sl','pnl','trades'])
    # csvfile.to_csv(f'sample_result.csv',index=False)
    for r in result[:15]:
        if r[6] > cur_max:
            cur_max = r[6]
        if r[6] > 0:
            print('\n', r[0],r[1],r[2],r[3],r[4],r[5],r[6],f'cur_max: {cur_max}',r[7])
    total = np.append(total,result[:10])
    np.save('total.npy', total)
    return total
# np.save(total, 'total.npy')

if __name__ == '__main__':

    # apatsl = list(itertools.product([x for x in range(2,20)],[x / 10 for x in range(1,20)], [x / 10 for x in range(2,20)]))
    apatsl = [[2,0.5,2.4]]
    n = 20
    r1r2r3 = list(itertools.product([x / n for x in range(n)],[x / n for x in range(n)],[x / n for x in range(n)]))
    # with Pool(processes = 10) as pool:
    #     resultss = pool.map(lambda cp:func(cp), cpap)
    resultss = []
    r = []
    for c in apatsl:
        r = func(c,r1r2r3,r,code = 'IM')
    resultss = np.append(resultss,r)
    np.save('resultss.npy',resultss)