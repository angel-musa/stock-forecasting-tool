import plotly.graph_objs as go
import pandas_ta as ta

def plot_indicators(fig, df, indicator):
    if indicator == 'MACD':
        macd = df.ta.macd()
        fig.add_trace(go.Scatter(x=macd.index, y=macd['MACD_12_26_9'], mode='lines', name='MACD'), secondary_y=True)
        fig.add_trace(go.Scatter(x=macd.index, y=macd['MACDs_12_26_9'], mode='lines', name='Signal'), secondary_y=True)
        fig.add_trace(go.Bar(x=macd.index, y=macd['MACDh_12_26_9'], name='Histogram'), secondary_y=True)
    elif indicator == 'RSI':
        rsi = df.ta.rsi()
        fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode='lines', name='RSI'), secondary_y=True)
    elif indicator == 'BBANDS':
        bbands = df.ta.bbands()
        fig.add_trace(go.Scatter(x=bbands.index, y=bbands['BBU_5_2.0'], mode='lines', name='Upper Band'), secondary_y=True)
        fig.add_trace(go.Scatter(x=bbands.index, y=bbands['BBM_5_2.0'], mode='lines', name='Middle Band'), secondary_y=False)
        fig.add_trace(go.Scatter(x=bbands.index, y=bbands['BBL_5_2.0'], mode='lines', name='Lower Band'), secondary_y=True)
    elif indicator == 'SMA':
        sma = df.ta.sma()
        fig.add_trace(go.Scatter(x=sma.index, y=sma, mode='lines', name='SMA'), secondary_y=True)
    elif indicator == 'EMA':
        ema = df.ta.ema()
        fig.add_trace(go.Scatter(x=ema.index, y=ema, mode='lines', name='EMA'), secondary_y=True)
    elif indicator == 'PE_RATIO':
        pe_ratio = df['Close'] / df['Earnings']  # Assuming earnings data is available
        fig.add_trace(go.Scatter(x=pe_ratio.index, y=pe_ratio, mode='lines', name='PE Ratio'), secondary_y=True)

    return fig
