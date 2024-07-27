from dash import dcc, html
import dash_bootstrap_components as dbc
from data_loader import load_tickers

def create_layout(app):
    tickers = load_tickers()

    return html.Div(
        className="container",
        children=[
            # Header and logo
            html.Div(
                className="header", 
                children=[
                    html.Div(className="logo-container", children=[
                        html.Img(src=app.get_asset_url('logo2.png'), alt='Logo', className='logo')
                    ]),
                    html.H1("Fortuna", className="title") 
                ]
            ),
            # Controls and dropdown
            html.Div(className="header-row", children=[ 
                html.Div(className="buttons", children=[
                    dbc.Button("Edit Filters", id='edit-filters-button', color='primary', n_clicks=0, className="button-filters"),
                    dbc.Button("Indicators", id='indicators-button', color='primary', n_clicks=0, className="button-indicators"),
                    dbc.Button("Graphs", id='graphs-button', color='primary', n_clicks=0, className="button-graphs"),
                    dbc.Button("Forecast", id="forecast-button", className="ms-auto")
                ]),
                html.Div(className="dropdown-container", children=[
                    dcc.Dropdown(
                        id='stock-dropdown',
                        options=[{'label': ticker, 'value': ticker} for ticker in tickers],
                        value=None,  
                        placeholder='Enter a ticker symbol',
                        className="dropdown"
                    )
                ]),
            ]),
            # Graph
            dcc.Graph(id='stock-graph', className="graph-container"),
            # About Section
            html.Div(className="about-section", children=[
                dcc.Tabs(
                    id="tabs",
                    value="overview-tab",  # Default tab
                    children=[
                        dcc.Tab(label='Overview', value='overview-tab'),
                        dcc.Tab(label='Financials', value='financials-tab'),
                        dcc.Tab(label='Latest News', value='news-tab'),
                    ]
                ),
                html.Div(id='tabs-content')  
            ]),
            # Modals
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Edit Filters")),
                    dbc.ModalBody(
                        html.Div([
                            html.Label("Select Time Period:", className="mb-2"),
                            dcc.RadioItems(
                                id='time-period-radio',
                                options=[
                                    {'label': '1 Day (1d)', 'value': '1d'},
                                    {'label': '5 Days (5d)', 'value': '5d'},
                                    {'label': '1 Month (1mo)', 'value': '1mo'},
                                    {'label': '3 Months (3mo)', 'value': '3mo'},
                                    {'label': '6 Months (6mo)', 'value': '6mo'},
                                    {'label': '1 Year (1y)', 'value': '1y'},
                                    {'label': '2 Years (2y)', 'value': '2y'},
                                    {'label': '5 Years (5y)', 'value': '5y'},
                                    {'label': '10 Years (10y)', 'value': '10y'},
                                    {'label': 'Year-to-Date (ytd)', 'value': 'ytd'},
                                    {'label': 'Max', 'value': 'max'},
                                ],
                                value='1y',  # Set default value to 1 year
                                className="d-grid gap-2 col-6 mx-auto"
                            ),
                        ])
                    ),
                    dbc.ModalFooter([
                        dbc.Button("Apply", id="apply-filters", className="ms-auto"),
                        dbc.Button("Close", id="close-filters", className="ms-auto")
                    ]),
                ],
                id="modal-filters",
                is_open=False,
                size="lg",
            ),
            # Modal for "Indicators"
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Indicators")),
                    dbc.ModalBody(
                        html.Div([
                            html.Label("Select Technical Indicator:", className="mb-2"),
                            dcc.RadioItems(
                                id='indicators-radio',
                                options=[
                                    {'label': 'Moving Average Convergence Divergence (MACD)', 'value': 'MACD'},
                                    {'label': 'Relative Strength Index (RSI)', 'value': 'RSI'},
                                    {'label': 'Bollinger Bands (BBANDS)', 'value': 'BBANDS'},
                                    {'label': 'Simple Moving Average (SMA)', 'value': 'SMA'},
                                    {'label': 'Exponential Moving Average (EMA)', 'value': 'EMA'},
                                    {'label': 'PE Ratio', 'value': 'PE_RATIO'}
                                ],
                                value=None,
                                className="d-grid gap-2 col-6 mx-auto"
                            ),
                        ])
                    ),
                    dbc.ModalFooter([
                        dbc.Button("Apply", id="apply-indicators", className="ms-auto"),
                        dbc.Button("Close", id="close-indicators", className="ms-auto")
                    ]),
                ],
                id="modal-indicators",
                is_open=False,
                size="lg",
            ),
            # Modal for "Graphs"
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Graphs")),
                    dbc.ModalBody(
                        html.Div([
                            html.Label("Select Graph Type:", className="mb-2"),
                            dcc.RadioItems(
                                id='graphs-radio',
                                options=[
                                    {'label': 'Line', 'value': 'Line'},
                                    {'label': 'Candlestick', 'value': 'Candlestick'},
                                    {'label': 'OHLC', 'value': 'OHLC'},
                                    {'label': 'Bar', 'value': 'Bar'},
                                    {'label': 'Scatter', 'value': 'Scatter'}
                                ],
                                value='line',  # Set default value to 1 year
                                className="d-grid gap-2 col-6 mx-auto"
                            ),
                        ])
                    ), 
                    dbc.ModalFooter([
                        dbc.Button("Apply", id="apply-graphs", className="ms-auto"),
                        dbc.Button("Close", id="close-graphs", className="ms-auto")
                    ]),
                ],
                id="modal-graphs",
                is_open=False,
                size="lg",
            ),
        ]
    )


