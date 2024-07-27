import yfinance as yf
import pandas as pd
from prophet import Prophet
import xgboost as xgb
from datetime import datetime, timedelta
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
import plotly.graph_objects as go
import optuna
from sklearn.metrics import mean_squared_error

#NOTE: same as mixed-ratio.py but different name!

def generate_forecast(ticker): 
    # Define the function to add technical indicators
    def add_technical_indicators(df):
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df = df.dropna()  # Drop NaNs created by rolling functions
        df = df[df['Volatility'] > 0]  # Ensure positive volatility
        return df

    # Function to prepare data for Prophet
    def prepare_data(ticker):
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = yf.download(ticker, start='2021-01-01', end=end_date)
        data = data[data['Close'] > 0]  # Ensure there are no negative or zero closing prices
        df = pd.DataFrame()
        df['ds'] = data.index
        df['y'] = data['Close'].values
        return df, data

    # Function to fit Prophet model and forecast
    def fit_and_forecast_prophet(train_data, test_data, params):
        model = Prophet(**params)
        model.fit(train_data)
        forecast = model.predict(test_data)
        return model, forecast

    # Function to add Prophet features
    def add_prophet_features(df, forecast):
        df = df.copy()
        df = df.merge(forecast[['ds', 'trend', 'yhat']], on='ds', how='left')
        return df

    # Define the objective function for Bayesian Optimization
    def objective2(trial, train_data_with_features, train_data):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1.0),
            'gamma': trial.suggest_uniform('gamma', 0, 1),
            'lambda': trial.suggest_loguniform('lambda', 1e-4, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-4, 1.0)
        }
        
        model = xgb.XGBRegressor(**params)
        
        # Train the model
        model.fit(train_data_with_features[['trend']], train_data['y'])
        
        # Predict
        predictions = model.predict(train_data_with_features['trend'])
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(train_data_with_features['y'], predictions))
        return rmse


    def optimize_xgb(train_data_with_features, train_data): 
        study = optuna.create_study(direction = 'minimize')
        func = lambda trial: objective2(trial, train_data_with_features, train_data) # create partial function to pass the dataframes 
        study.optimize(func, n_trials =3)
        return study.best_params

    # # Function to fit XGBoost model
    # def fit_xgboost(train_data_with_features, train_data):
    #     #trains xgboost on training data that has been enriched with prophet features 
    #     #specifically uses trend feature from Prophet as predictor for actual target values 
    #     best_params = optimize_xgb(train_data_with_features, train_data)
    #     xgb_model = xgb.XGBRegressor(**best_params)
    #     xgb_model.fit(train_data_with_features[['trend']], train_data['y'])
    #     #returns trained xgb model 
    #     return xgb_model

    # Function to fit XGBoost model
    def fit_xgboost(train_data_with_features, train_data):
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1)
        xgb_model.fit(train_data_with_features[['trend']], train_data['y'])
        return xgb_model


    # Function to combine predictions
    def combine_predictions(test_data_with_features):
        test_data_with_features['combined_pred'] = (test_data_with_features['yhat'] + test_data_with_features['xgb_pred']) / 2
        test_data_with_features['average_pred'] = (test_data_with_features['yhat'] + test_data_with_features['combined_pred']) / 2
        return test_data_with_features

    # Function to calculate the mean price difference
    def calculate_mean_price_diff(params, split_ratio):
        changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale, n_changepoints, changepoint_range, seasonality_mode = params
        train_size = int(len(df) * split_ratio)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:].copy()

        # Fit Prophet and forecast
        params_dict = {
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'holidays_prior_scale': holidays_prior_scale,
            'n_changepoints': int(n_changepoints),
            'changepoint_range': changepoint_range,
            'seasonality_mode': 'additive' if seasonality_mode < 0.5 else 'multiplicative'
        }
        prophet_model, forecast_test = fit_and_forecast_prophet(train_data, test_data, params_dict)

        # Add Prophet features to train and test data
        train_data_with_features = add_prophet_features(train_data, prophet_model.predict(train_data))
        test_data_with_features = add_prophet_features(test_data, forecast_test)

        # Fit XGBoost and make predictions
        xgb_model = fit_xgboost(train_data_with_features, train_data)
        test_data_with_features['xgb_pred'] = xgb_model.predict(test_data_with_features[['trend']])

        # Combine predictions
        test_data_with_features = combine_predictions(test_data_with_features)

        # Forecast the next 5 days
        forecast_end_date = df['ds'].max()
        future_dates = pd.date_range(start=forecast_end_date + pd.Timedelta(days=1), periods=5, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        forecast_5days = prophet_model.predict(future_df)
        future_df_with_features = add_prophet_features(future_df, forecast_5days)
        future_df_with_features['xgb_pred'] = xgb_model.predict(future_df_with_features[['trend']])
        future_df_with_features = combine_predictions(future_df_with_features)

        # Calculate the mean price of the next 5 days and compare it to the last closing price
        mean_forecast_price = future_df_with_features['average_pred'].mean()
        mean_diff = abs(mean_forecast_price - last_closing_price)

        return mean_diff

    df, data = prepare_data(ticker)

    last_closing_price = df['y'].iloc[-1]

    # Define the bounds for Bayesian Optimization
    search_space = [
        Real(0.001, 0.05, name='changepoint_prior_scale'),
        Real(1.0, 10.0, name='seasonality_prior_scale'),
        Real(1.0, 10.0, name='holidays_prior_scale'),
        Integer(10, 50, name='n_changepoints'),
        Real(0.6, 0.9, name='changepoint_range'),
        Real(0, 1, name='seasonality_mode')  # 0 for additive, 1 for multiplicative
    ]

    split_ratios = [0.66, 0.75]
    best_split_ratio = None
    min_difference = float('inf')
    best_forecast_next_day = None
    best_params = None

    # Iterate over each split ratio
    for split_ratio in split_ratios:
        # Define the objective function for Bayesian Optimization
        def objective(params):
            return calculate_mean_price_diff(params, split_ratio)

        # Perform Bayesian Optimization
        res = gp_minimize(objective, search_space, n_calls=20, random_state=42)
        
        # Extract best parameters
        current_best_params = {
            'changepoint_prior_scale': res.x[0],
            'seasonality_prior_scale': res.x[1],
            'holidays_prior_scale': res.x[2],
            'n_changepoints': res.x[3],
            'changepoint_range': res.x[4],
            'seasonality_mode': 'additive' if res.x[5] < 0.5 else 'multiplicative'
        }

        # Fit Prophet and forecast with best parameters
        prophet_model, forecast_test = fit_and_forecast_prophet(df.iloc[:int(len(df) * split_ratio)], df.iloc[int(len(df) * split_ratio):], current_best_params)

        # Add Prophet features to train and test data
        train_data_with_features = add_prophet_features(df.iloc[:int(len(df) * split_ratio)], prophet_model.predict(df.iloc[:int(len(df) * split_ratio)]))
        test_data_with_features = add_prophet_features(df.iloc[int(len(df) * split_ratio):], forecast_test)

        # Fit XGBoost and make predictions
        xgb_model = fit_xgboost(train_data_with_features, df.iloc[:int(len(df) * split_ratio)])
        test_data_with_features['xgb_pred'] = xgb_model.predict(test_data_with_features[['trend']])

        # Combine predictions
        test_data_with_features = combine_predictions(test_data_with_features)

        # Forecast the next 5 days
        forecast_end_date = df['ds'].max()
        future_dates = pd.date_range(start=forecast_end_date + pd.Timedelta(days=1), periods=5, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        forecast_5days = prophet_model.predict(future_df)
        future_df_with_features = add_prophet_features(future_df, forecast_5days)
        future_df_with_features['xgb_pred'] = xgb_model.predict(future_df_with_features[['trend']])
        future_df_with_features = combine_predictions(future_df_with_features)

        # Calculate the mean price of the next 5 days and compare it to the last closing price
        mean_forecast_price = future_df_with_features['average_pred'].mean()
        mean_diff = abs(mean_forecast_price - last_closing_price)

        # Update the best split ratio if the current one is better
        if mean_diff < min_difference:
            min_difference = mean_diff
            best_split_ratio = split_ratio
            best_forecast_next_day = future_df_with_features['average_pred'].values
            best_params = current_best_params

    # Print the best split ratio and the forecasted next day's price
    print(f"Best split ratio: {best_split_ratio}")
    print(f"Mean of forecasted 5-day prices: {best_forecast_next_day.mean()}")

    # Use the best split ratio for final training and forecasting
    train_size = int(len(df) * best_split_ratio)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:].copy()

    # Fit Prophet and forecast with best split ratio and parameters
    prophet_model, forecast_test = fit_and_forecast_prophet(train_data, test_data, best_params)

    # Add Prophet features to train and test data
    train_data_with_features = add_prophet_features(train_data, prophet_model.predict(train_data))
    test_data_with_features = add_prophet_features(test_data, forecast_test)

    # Fit XGBoost and make predictions
    xgb_model = fit_xgboost(train_data_with_features, train_data)
    test_data_with_features['xgb_pred'] = xgb_model.predict(test_data_with_features[['trend']])

    # Combine predictions
    test_data_with_features = combine_predictions(test_data_with_features)

    # Forecast the next 7 days and for the rest of 2024
    forecast_end_date = df['ds'].max()
    future_dates = pd.date_range(start=forecast_end_date + pd.Timedelta(days=1), periods=7, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    forecast_7days = prophet_model.predict(future_df)
    future_df_with_features = add_prophet_features(future_df, forecast_7days)
    future_df_with_features['xgb_pred'] = xgb_model.predict(future_df_with_features[['trend']])
    future_df_with_features = combine_predictions(future_df_with_features)

    # Forecast for 2024
    future_dates_2024 = pd.date_range(start='2024-07-22', end='2024-12-31', freq='D')
    future_df_2024 = pd.DataFrame({'ds': future_dates_2024})
    forecast_2024 = prophet_model.predict(future_df_2024)
    future_df_with_features_2024 = add_prophet_features(future_df_2024, forecast_2024)
    future_df_with_features_2024['xgb_pred'] = xgb_model.predict(future_df_with_features_2024[['trend']])
    future_df_with_features_2024 = combine_predictions(future_df_with_features_2024)

    return future_df_with_features_2024
    # # Plot the results
    # fig = go.Figure()

    # # # Actual closing prices
    # # fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))

    # # # Prophet-only predictions
    # # fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data_with_features['yhat'], mode='lines', name='Prophet Only'))

    # # # XGBoost-only predictions
    # # fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data_with_features['xgb_pred'], mode='lines', name='XGBoost Only'))

    # # # Combined predictions
    # # fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data_with_features['combined_pred'], mode='lines', name='Combined Predictions'))

    # # # Average predictions
    # # fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data_with_features['average_pred'], mode='lines', name='Average Predictions'))

    # # # Forecast for the next 7 days
    # # fig.add_trace(go.Scatter(x=future_df_with_features['ds'], y=future_df_with_features['average_pred'], mode='lines', name='7 Day Forecast'))

    # # Forecast for the rest of 2024
    # fig.add_trace(go.Scatter(x=future_df_with_features_2024['ds'], y=future_df_with_features_2024['average_pred'], mode='lines', name='2024 Forecast'))

    # # # Add titles and labels
    # # fig.update_layout(
    # #     title= ticker + 'Stock Price Forecast',
    # #     xaxis_title='Date',
    # #     yaxis_title='Price',
    # #     hovermode='x unified'
    # # )

    # return fig

