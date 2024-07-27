from dash import html, dcc
import yfinance as yf

def get_tab_content(tab, df, selected_stock):
    info = yf.Ticker(selected_stock).info

    if tab == 'overview-tab':
        overview_content = html.Div([
            html.P(f"Sector: {info.get('sector', 'N/A')}"),
            html.P(f"Industry: {info.get('industry', 'N/A')}"),
            html.P(f"Website: {info.get('website', 'N/A')}"),
            dcc.Markdown(f"**Company Overview:**\n\n{info.get('longBusinessSummary', 'N/A')}")
        ])
        return overview_content
    elif tab == 'financials-tab':
        financials = yf.Ticker(selected_stock).financials
        financials = financials.iloc[:, -1]  # Get most recent period

        # Filter out object datatypes (like dates) and NaNs
        financials_filtered = financials.loc[financials.notnull() & ~financials.apply(lambda x: isinstance(x, str))]

        # Create html.P elements for display
        financials_content = [html.H3("Key Financials:")] + \
                             [html.P(f"{index}: {value:,.0f}") for index, value in financials_filtered.items()]

        return html.Div(financials_content)
    elif tab == 'news-tab':
        news_data = yf.Ticker(selected_stock).news
        if (news_data) and (len(news_data) > 0):
            news_list = [
                html.Li(
                    html.A(
                        f"{article['title']}",
                        href=article['link'],
                        target="_blank"
                    )
                )
                for article in news_data[:7]  
            ]
            return html.Ul(news_list)
        else:
            return html.Div("No news available for this stock.")
    else:
        return html.Div([
            html.H3("Overview"),
            html.P("Provide an overview of the stock, including its current market position, recent news, and historical performance."),
        ])
