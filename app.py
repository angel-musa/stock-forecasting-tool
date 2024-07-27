import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from layout import create_layout
from callbacks import register_callbacks

# Initialize Dash app
app = dash.Dash(__name__, assets_folder='assets', external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = create_layout(app)

# State to store the selected graph type
app.layout.children.append(
    dcc.Store(id='graph-type-store', data='Line'), # Store initial graph type as 'Line'
)

# State to store the selected graph type
app.layout.children.append(
    dcc.Store(id='indicators-store', data=None)
)

# Register callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)
