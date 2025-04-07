import pandas as pd
from typing import Optional
from datetime import timedelta
from plotly.express import line
def plot_one_sample(features:pd.DataFrame, targets:pd.Series,example_id:int,prediction:Optional[pd.Series]=None):
    """
    Plot one sample of the features and targets
    """
    features_ = features.iloc[example_id]
    target_ = targets.iloc[example_id]
    ts_columns = [col for col in features.columns if col.startswith('rides_previous_')]
    ts_values = [features_[col] for col in ts_columns] + [target_]

    ts_dates = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_values)-1),
        features_['pickup_hour'],
        freq='H'
    )
    # line plot with past values
    title = f'pickup_hour: {features_["pickup_hour"]} target: {target_} location_id: {features_["pickup_location_id"]}'
    print('title', title)
    
    fig = line(
        x=ts_dates,
        y=ts_values,
        title=title,
        markers=True
    )

    # green dot for the value we wanna predict
    fig.add_scatter(
        x=[ts_dates[-1] + timedelta(hours=1)],
        y=[target_],
        mode='markers',
        marker=dict(color='green', size=10),
        name='actual value'
    )

    # red dot for the prediction
    if prediction is not None:
        fig.add_scatter(
            x=[ts_dates[-1] + timedelta(hours=1)],
            y=[prediction],
            mode='markers',
            marker=dict(color='red', size=10),
            name='prediction'
        )

    return fig
