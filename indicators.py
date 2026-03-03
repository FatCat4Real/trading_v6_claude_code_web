import numpy as np
import pandas as pd
import talib


# --- Helpers ---

def crossover(a, b):
    """a crosses above b"""
    return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False)

def crossunder(a, b):
    """a crosses below b"""
    return ((a < b) & (a.shift(1) >= b.shift(1))).fillna(False)

def const(val, index):
    return pd.Series(val, index=index)


# ============================================================
# TradingView Official Built-in Indicators (22)
# ============================================================

def macd(df):
    m, s, h = talib.MACD(df['CLOSE'].values, 12, 26, 9)
    m, s = pd.Series(m, index=df.index), pd.Series(s, index=df.index)
    return crossover(m, s), crossunder(m, s)

def rsi(df):
    r = pd.Series(talib.RSI(df['CLOSE'].values, 14), index=df.index)
    return crossover(r, const(30, df.index)), crossunder(r, const(70, df.index))

def stochastic(df):
    k, d = talib.STOCH(df['HIGH'].values, df['LOW'].values, df['CLOSE'].values,
                        14, 3, 0, 3, 0)
    k, d = pd.Series(k, index=df.index), pd.Series(d, index=df.index)
    buy = crossover(k, d) & (k < 20)
    sell = crossunder(k, d) & (k > 80)
    return buy.fillna(False), sell.fillna(False)

def stoch_rsi(df):
    fastk, fastd = talib.STOCHRSI(df['CLOSE'].values, 14, 14, 3, 0)
    k, d = pd.Series(fastk, index=df.index), pd.Series(fastd, index=df.index)
    buy = crossover(k, d) & (k < 20)
    sell = crossunder(k, d) & (k > 80)
    return buy.fillna(False), sell.fillna(False)

def bollinger_bands(df):
    upper, middle, lower = talib.BBANDS(df['CLOSE'].values, 20, 2, 2, 0)
    close = df['CLOSE']
    upper = pd.Series(upper, index=df.index)
    lower = pd.Series(lower, index=df.index)
    # mean reversion: buy on recovery from lower band, sell on reaching upper band
    return crossover(close, lower), crossover(close, upper)

def sma_cross(df):
    short = pd.Series(talib.SMA(df['CLOSE'].values, 10), index=df.index)
    long = pd.Series(talib.SMA(df['CLOSE'].values, 30), index=df.index)
    return crossover(short, long), crossunder(short, long)

def ema_cross(df):
    short = pd.Series(talib.EMA(df['CLOSE'].values, 10), index=df.index)
    long = pd.Series(talib.EMA(df['CLOSE'].values, 30), index=df.index)
    return crossover(short, long), crossunder(short, long)

def cci_signal(df):
    c = pd.Series(talib.CCI(df['HIGH'].values, df['LOW'].values, df['CLOSE'].values, 20),
                  index=df.index)
    return crossover(c, const(-100, df.index)), crossunder(c, const(100, df.index))

def adx_di(df):
    plus_di = pd.Series(talib.PLUS_DI(df['HIGH'].values, df['LOW'].values,
                                       df['CLOSE'].values, 14), index=df.index)
    minus_di = pd.Series(talib.MINUS_DI(df['HIGH'].values, df['LOW'].values,
                                         df['CLOSE'].values, 14), index=df.index)
    return crossover(plus_di, minus_di), crossunder(plus_di, minus_di)

def awesome_oscillator(df):
    median = (df['HIGH'] + df['LOW']) / 2
    ao = (pd.Series(talib.SMA(median.values, 5), index=df.index)
          - pd.Series(talib.SMA(median.values, 34), index=df.index))
    return crossover(ao, const(0, df.index)), crossunder(ao, const(0, df.index))

def momentum_signal(df):
    mom = pd.Series(talib.MOM(df['CLOSE'].values, 10), index=df.index)
    return crossover(mom, const(0, df.index)), crossunder(mom, const(0, df.index))

def williams_r(df):
    wr = pd.Series(talib.WILLR(df['HIGH'].values, df['LOW'].values,
                                df['CLOSE'].values, 14), index=df.index)
    # leaving oversold (crosses above -80), leaving overbought (crosses below -20)
    return crossover(wr, const(-80, df.index)), crossunder(wr, const(-20, df.index))

def ichimoku(df):
    high, low = df['HIGH'], df['LOW']
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    return crossover(tenkan, kijun), crossunder(tenkan, kijun)

def parabolic_sar(df):
    sar = pd.Series(talib.SAR(df['HIGH'].values, df['LOW'].values, 0.02, 0.2),
                    index=df.index)
    close = df['CLOSE']
    return crossover(close, sar), crossunder(close, sar)

def mfi_signal(df):
    m = pd.Series(talib.MFI(df['HIGH'].values, df['LOW'].values,
                             df['CLOSE'].values, df['TOTAL_VOLUME'].astype(float).values, 14),
                  index=df.index)
    return crossover(m, const(20, df.index)), crossunder(m, const(80, df.index))

def cmf_signal(df):
    high, low, close = df['HIGH'], df['LOW'], df['CLOSE']
    vol = df['TOTAL_VOLUME'].astype(float)
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * vol
    cmf = mfv.rolling(20).sum() / vol.rolling(20).sum()
    return crossover(cmf, const(0, df.index)), crossunder(cmf, const(0, df.index))

def obv_signal(df):
    o = pd.Series(talib.OBV(df['CLOSE'].values, df['TOTAL_VOLUME'].astype(float).values),
                  index=df.index)
    sma = o.rolling(20).mean()
    return crossover(o, sma), crossunder(o, sma)

def aroon_signal(df):
    down, up = talib.AROON(df['HIGH'].values, df['LOW'].values, 14)
    up = pd.Series(up, index=df.index)
    down = pd.Series(down, index=df.index)
    return crossover(up, down), crossunder(up, down)

def roc_signal(df):
    r = pd.Series(talib.ROC(df['CLOSE'].values, 12), index=df.index)
    return crossover(r, const(0, df.index)), crossunder(r, const(0, df.index))

def supertrend(df):
    period, multiplier = 10, 3.0
    high, low, close = df['HIGH'].values, df['LOW'].values, df['CLOSE'].values
    atr = talib.ATR(high, low, close, period)
    hl2 = (high + low) / 2

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)  # 1=up, -1=down

    # find first valid index
    start = period  # ATR needs 'period' bars
    if start >= n:
        d = pd.Series(direction, index=df.index)
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    final_upper[start] = upper[start]
    final_lower[start] = lower[start]
    direction[start] = 1 if close[start] > lower[start] else -1

    for i in range(start + 1, n):
        # final lower band
        if lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        # final upper band
        if upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        # direction
        if direction[i-1] == 1:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close[i] > final_upper[i] else -1

    d = pd.Series(direction, index=df.index)
    buy = (d == 1) & (d.shift(1) == -1)
    sell = (d == -1) & (d.shift(1) == 1)
    return buy.fillna(False), sell.fillna(False)

def donchian_channel(df):
    period = 20
    upper = df['HIGH'].rolling(period).max().shift(1)
    lower = df['LOW'].rolling(period).min().shift(1)
    close = df['CLOSE']
    return crossover(close, upper), crossunder(close, lower)

def keltner_channel(df):
    period, mult = 20, 1.5
    ema = pd.Series(talib.EMA(df['CLOSE'].values, period), index=df.index)
    atr = pd.Series(talib.ATR(df['HIGH'].values, df['LOW'].values,
                               df['CLOSE'].values, period), index=df.index)
    upper = ema + mult * atr
    lower = ema - mult * atr
    close = df['CLOSE']
    return crossover(close, upper), crossunder(close, lower)


# ============================================================
# Additional Indicators (8)
# ============================================================

def hull_ma_cross(df):
    # Alan Hull's Hull Moving Average
    def hma(series, period):
        half = max(int(period / 2), 1)
        sqrt_p = max(int(np.sqrt(period)), 1)
        wma_half = pd.Series(talib.WMA(series, half), index=df.index)
        wma_full = pd.Series(talib.WMA(series, period), index=df.index)
        diff = 2 * wma_half - wma_full
        return pd.Series(talib.WMA(diff.values, sqrt_p), index=df.index)

    short = hma(df['CLOSE'].values, 9)
    long = hma(df['CLOSE'].values, 21)
    return crossover(short, long), crossunder(short, long)

def trix_signal(df):
    # Jack Hutson's TRIX
    t = pd.Series(talib.TRIX(df['CLOSE'].values, 15), index=df.index)
    signal = t.rolling(9).mean()
    return crossover(t, signal), crossunder(t, signal)

def dema_cross(df):
    # Patrick Mulloy's Double EMA
    short = pd.Series(talib.DEMA(df['CLOSE'].values, 10), index=df.index)
    long = pd.Series(talib.DEMA(df['CLOSE'].values, 30), index=df.index)
    return crossover(short, long), crossunder(short, long)

def tema_cross(df):
    # Patrick Mulloy's Triple EMA
    short = pd.Series(talib.TEMA(df['CLOSE'].values, 10), index=df.index)
    long = pd.Series(talib.TEMA(df['CLOSE'].values, 30), index=df.index)
    return crossover(short, long), crossunder(short, long)

def dpo_signal(df):
    # Detrended Price Oscillator
    period = 20
    shift = period // 2 + 1
    sma = pd.Series(talib.SMA(df['CLOSE'].values, period), index=df.index)
    d = df['CLOSE'] - sma.shift(shift)
    return crossover(d, const(0, df.index)), crossunder(d, const(0, df.index))

def ultimate_oscillator(df):
    # Larry Williams' Ultimate Oscillator
    uo = pd.Series(talib.ULTOSC(df['HIGH'].values, df['LOW'].values,
                                 df['CLOSE'].values, 7, 14, 28), index=df.index)
    return crossover(uo, const(30, df.index)), crossunder(uo, const(70, df.index))

def chande_momentum(df):
    # Tushar Chande's CMO
    cmo = pd.Series(talib.CMO(df['CLOSE'].values, 14), index=df.index)
    return crossover(cmo, const(0, df.index)), crossunder(cmo, const(0, df.index))

def linreg_slope(df):
    # Linear Regression Slope
    slope = pd.Series(talib.LINEARREG_SLOPE(df['CLOSE'].values, 14), index=df.index)
    return crossover(slope, const(0, df.index)), crossunder(slope, const(0, df.index))


# ============================================================
# Registry: {display_name: (function, source)}
# ============================================================

INDICATORS = {
    # TradingView official built-in indicators
    'MACD': (macd, 'TradingView'),
    'RSI': (rsi, 'TradingView'),
    'Stochastic': (stochastic, 'TradingView'),
    'Stochastic RSI': (stoch_rsi, 'TradingView'),
    'Bollinger Bands': (bollinger_bands, 'TradingView'),
    'SMA Cross': (sma_cross, 'TradingView'),
    'EMA Cross': (ema_cross, 'TradingView'),
    'CCI': (cci_signal, 'TradingView'),
    'ADX/DI': (adx_di, 'TradingView'),
    'Awesome Oscillator': (awesome_oscillator, 'TradingView'),
    'Momentum': (momentum_signal, 'TradingView'),
    'Williams %R': (williams_r, 'TradingView'),
    'Ichimoku': (ichimoku, 'TradingView'),
    'Parabolic SAR': (parabolic_sar, 'TradingView'),
    'MFI': (mfi_signal, 'TradingView'),
    'CMF': (cmf_signal, 'TradingView'),
    'OBV': (obv_signal, 'TradingView'),
    'Aroon': (aroon_signal, 'TradingView'),
    'ROC': (roc_signal, 'TradingView'),
    'Supertrend': (supertrend, 'TradingView'),
    'Donchian Channel': (donchian_channel, 'TradingView'),
    'Keltner Channel': (keltner_channel, 'TradingView'),
    # Additional indicators
    'Hull MA Cross': (hull_ma_cross, 'Alan Hull'),
    'TRIX': (trix_signal, 'ta-lib / Jack Hutson'),
    'DEMA Cross': (dema_cross, 'ta-lib / Patrick Mulloy'),
    'TEMA Cross': (tema_cross, 'ta-lib / Patrick Mulloy'),
    'DPO': (dpo_signal, 'ta-lib'),
    'Ultimate Oscillator': (ultimate_oscillator, 'ta-lib / Larry Williams'),
    'Chande Momentum': (chande_momentum, 'ta-lib / Tushar Chande'),
    'LinReg Slope': (linreg_slope, 'ta-lib'),
}
