import pandas as pd
from prophet import Prophet
from datetime import timedelta

def run_forecasting(df, start_date, end_date):
    days = (end_date - start_date).days + 1
    hourly_slots = [str(i) for i in range(1, 25)]
    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = df.columns.astype(str)

    forecast_results = {}

    for slot in hourly_slots:
        if slot not in df.columns:
            df[slot] = 0

    slot_forecasts = {}

    for slot in hourly_slots:
        slot_df = df[['Date', slot]].rename(columns={'Date': 'ds', slot: 'y'})
        slot_df['floor'] = 0
        slot_df = slot_df.dropna()

        if len(slot_df) < 8:
            continue

        train_df = slot_df[slot_df['ds'] <= start_date]

        model = Prophet()
        model.fit(train_df)

        future_dates = pd.date_range(start=start_date, periods=days)
        future_df = pd.DataFrame({'ds': future_dates})
        future_df['floor'] = 0

        forecast = model.predict(future_df)[['ds', 'yhat']]
        forecast.columns = ['Date', slot]
        forecast[slot] = forecast[slot].clip(lower=0).round()
        forecast.set_index('Date', inplace=True)

        slot_forecasts[slot] = forecast

    if not slot_forecasts:
        return pd.DataFrame(columns=['Date'] + hourly_slots)

    combined_forecast = pd.concat(
        [slot_forecasts.get(slot, pd.DataFrame(index=future_dates, columns=[slot]))
         for slot in hourly_slots],
        axis=1
    )

    combined_forecast.reset_index(inplace=True)
    combined_forecast.fillna(0, inplace=True)
    combined_forecast[hourly_slots] = combined_forecast[hourly_slots].astype(int)

    return combined_forecast
