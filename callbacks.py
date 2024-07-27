from dash import dcc, html, Input, Output, State, callback_context, no_update
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from data_loader import load_stock_data, load_tickers
from indicators import plot_indicators
from tabs import get_tab_content
import yfinance as yf
from forecasting import generate_forecast

tickers = load_tickers()
stock_data = load_stock_data(tickers)

def register_callbacks(app):
    @app.callback(
        [Output('stock-graph', 'figure'),
         Output('tabs-content', 'children')],
        [Input('stock-dropdown', 'value'),
         Input('tabs', 'value'),
         Input('apply-filters', 'n_clicks'),
         Input('apply-graphs', 'n_clicks'),
         Input('apply-indicators', 'n_clicks'),
         Input('forecast-button', 'n_clicks')],
        [State('time-period-radio', 'value'),
         State('graphs-radio', 'value'),
         State('graph-type-store', 'data'),
         State('indicators-radio', 'value')]
    )
    def update_graph(selected_stock, tab, n_clicks_filters, n_clicks_graphs, n_clicks_indicators, n_clicks_forecast, time_period, graph_type, current_graph_type, selected_indicator):
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]

        if selected_stock is None or selected_stock not in stock_data:
            return go.Figure(layout=dict(plot_bgcolor='white')), html.Div("Select a stock to see data.", className="empty-tab-message")

        df = stock_data[selected_stock]

        if changed_id == 'apply-filters.n_clicks':
            df = yf.download(selected_stock, period=time_period)

        if changed_id == 'forecast-button.n_clicks':
            future_df_with_features_2024 = generate_forecast(selected_stock)
            
            # If using make_subplots
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Close Price"))
            fig.add_trace(go.Scatter(x=future_df_with_features_2024['ds'], y=future_df_with_features_2024['average_pred'], mode='lines', name='2024 Forecast'))
            
            # Add titles and labels
            fig.update_layout(
                title=selected_stock + ' Stock Price Forecast',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified'
            )

            tab_content = get_tab_content(tab, df, selected_stock)
            return fig, tab_content

        # Use make_subplots if needed
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        if changed_id == 'apply-graphs.n_clicks' and graph_type:
            current_graph_type = graph_type

        if current_graph_type == 'Candlestick':
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candlestick"), secondary_y=False)
        elif current_graph_type == 'OHLC':
            fig.add_trace(go.Ohlc(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"), secondary_y=False)
        elif current_graph_type == 'Bar':
            fig.add_trace(go.Bar(x=df.index, y=df['Close'], name="Bar"), secondary_y=False)
        elif current_graph_type == 'Scatter':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='markers', name="Scatter"), secondary_y=False)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Close Price"), secondary_y=False)

        if changed_id == 'apply-indicators.n_clicks':
            fig = plot_indicators(fig, df, selected_indicator)

        fig.update_layout(
            title=f'{selected_stock} Price Data',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            legend_title='Legend'
        )

        tab_content = get_tab_content(tab, df, selected_stock)

        return fig, tab_content

    @app.callback(
        [Output('graph-type-store', 'data'),
         Output('indicators-store', 'data'),
         Output('modal-graphs', 'is_open'),
         Output('modal-indicators', 'is_open'),
         Output('modal-filters', 'is_open')],
        [Input('apply-graphs', 'n_clicks'),
         Input('apply-indicators', 'n_clicks'),
         Input('graphs-button', 'n_clicks'),
         Input('close-graphs', 'n_clicks'),
         Input('indicators-button', 'n_clicks'),
         Input('close-indicators', 'n_clicks'),
         Input('edit-filters-button', 'n_clicks'),
         Input('close-filters', 'n_clicks'),
         Input('apply-filters', 'n_clicks')],
        [State('graphs-radio', 'value'),
         State('indicators-radio', 'value'),
         State('time-period-radio', 'value'),
         State('modal-graphs', 'is_open'),
         State('modal-indicators', 'is_open'),
         State('modal-filters', 'is_open')]
    )
    def update_settings_and_toggle_modals(n_clicks_graphs, n_clicks_indicators, n_clicks_graphs_button, n_clicks_close_graphs, n_clicks_indicators_button, n_clicks_close_indicators, n_clicks_edit_filters, n_clicks_close_filters, n_clicks_apply_filters, graph_type, indicator, time_period, is_graph_modal_open, is_indicator_modal_open, is_filter_modal_open):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, is_graph_modal_open, is_indicator_modal_open, is_filter_modal_open

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'apply-graphs':
            return graph_type, no_update, False, is_indicator_modal_open, is_filter_modal_open
        elif triggered_id == 'apply-indicators':
            return no_update, indicator, is_graph_modal_open, False, is_filter_modal_open
        elif triggered_id == 'apply-filters':
            return no_update, no_update, is_graph_modal_open, is_indicator_modal_open, False
        elif triggered_id == 'graphs-button':
            return no_update, no_update, not is_graph_modal_open, is_indicator_modal_open, is_filter_modal_open
        elif triggered_id == 'close-graphs':
            return no_update, no_update, False, is_indicator_modal_open, is_filter_modal_open
        elif triggered_id == 'indicators-button':
            return no_update, no_update, is_graph_modal_open, not is_indicator_modal_open, is_filter_modal_open
        elif triggered_id == 'close-indicators':
            return no_update, no_update, is_graph_modal_open, False, is_filter_modal_open
        elif triggered_id == 'edit-filters-button':
            return no_update, no_update, is_graph_modal_open, is_indicator_modal_open, not is_filter_modal_open
        elif triggered_id == 'close-filters':
            return no_update, no_update, is_graph_modal_open, is_indicator_modal_open, False

        return no_update, no_update, is_graph_modal_open, is_indicator_modal_open, is_filter_modal_open
