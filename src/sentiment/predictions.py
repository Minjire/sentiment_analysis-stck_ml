import pandas as pd
from datetime import datetime


def get_sentiment_predictions(prediction_data):
    predictions_df = pd.read_csv(prediction_data)
    predictions_df.raw_predictions = predictions_df.raw_predictions.apply(eval)
    # convert column values to datetime
    predictions_df.Date = pd.to_datetime(predictions_df.Date)
    predictions_df.Date = predictions_df.Date.dt.strftime("%Y-%m-%d")
    # get today's date
    day_date = datetime.today().date()
    # get rows with today's news
    today_df = predictions_df[predictions_df['Date'] == day_date]
    # find average of raw predictions
    av_predictions = [sum(e) / len(e) for e in zip(*today_df.raw_predictions.values.tolist())]
    return av_predictions
