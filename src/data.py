import requests
from pathlib import Path
import pandas as pd
from paths import RAW_DATA_DIR
from typing import Optional,List
from tqdm import tqdm
import numpy as np

def download_raw_data(year:int,month:int)->Path:
    URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        open(path,'wb').write(response.content)
        return path
    else:
        raise Exception(f"{URL} is not available")
    

def validate_raw_data(
        rides:pd.DataFrame,
        year:int,
        month:int,
)->pd.DataFrame:
    this_month_start = f'{year}-{month:02d}-01'
    this_month_end = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < this_month_end]
    return rides

def load_raw_data(year:int,months:Optional[List[int]]=None)->pd.DataFrame:
    rides = pd.DataFrame()
    if months is None:
        # download data only for the months specified by `months`
        months = list(range(1,13))
    elif isinstance(months,int):
        # download data for the entire year all (all months)
        months = [months]
    
    for month in months:
        local_file = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        if not local_file.exists():
            try:
                print(f"Downloading data for {year}-{month:02d}...")
                download_raw_data(year,month)
            except:
                print(f"Error downloading data for {year}-{month:02d}")
                continue
        else:
            print(f"File already exists for {year}-{month:02d}")

        # Load and rename columns
        rides_one_month = pd.read_parquet(local_file)
        rides_one_month = rides_one_month[['tpep_pickup_datetime','PULocationID']]
        rides_one_month.rename(columns={'tpep_pickup_datetime':'pickup_datetime','PULocationID':'pickup_location_id'},inplace=True)
        
        # validate 
        rides_one_month = validate_raw_data(rides_one_month, year, month)  # Store validated data back in rides_one_month

        # append to existing data
        rides = pd.concat([rides, rides_one_month])

    # keep only time and origin of the ride
    rides = rides[['pickup_datetime','pickup_location_id']]

    return rides


def add_missing_rows(agg_rides:pd.DataFrame) -> pd.DataFrame:
    location_ids = agg_rides["pickup_location_id"].unique()
    full_range = pd.date_range(
        agg_rides["pickup_hour"].min(),
        agg_rides["pickup_hour"].max(),
        freq="H"
    )
    print(full_range)
    output = pd.DataFrame()
    for location_id in tqdm(location_ids):
        # keep only rides for this 'location_id' (filter operation)
        agg_rides_i = agg_rides.loc[agg_rides.pickup_location_id == location_id,['pickup_hour','rides_count']]

        # quick way to add missing dates with 0 in a series (reindex operation)
        agg_rides_i.set_index("pickup_hour", inplace=True)
        agg_rides_i.index = pd.to_datetime(agg_rides_i.index)
        agg_rides_i = agg_rides_i.reindex(full_range, fill_value=0)

        # add back the location_id column
        agg_rides_i["pickup_location_id"] = location_id

        output = pd.concat([output, agg_rides_i])
    
    output = output.reset_index().rename(columns={"index":"pickup_hour"})
    return output


def get_cutoff_indices(
        data: pd.DataFrame,
        n_features: int,
        step_size: int,
)->list:
    stop_position = len(data) - 1
    indices = []

    # start the first subsequence at index 0
    subsequence_start_index = 0
    subsequence_mid_index = n_features 
    subsequence_end_index = n_features + 1

    while subsequence_end_index <= stop_position:
        indices.append((subsequence_start_index, subsequence_mid_index, subsequence_end_index))
        subsequence_start_index += step_size
        subsequence_mid_index += step_size
        subsequence_end_index += step_size

    return indices


def transform_raw_data_into_ts_data(
        rides:pd.DataFrame,
)->pd.DataFrame:
    rides["pickup_hour"] = rides["pickup_datetime"].dt.floor("h")
    agg_rides = rides.groupby(["pickup_hour", "pickup_location_id"]).size().reset_index(name="rides_count")

    agg_rides = add_missing_rows(agg_rides)

    return agg_rides

def transform_raw_data_into_features_and_targets(
    ts_data:pd.DataFrame,
    input_seq_length:int,
    step_size:int,
)->pd.DataFrame:
    assert set(ts_data.columns) == {"pickup_hour","rides_count","pickup_location_id"}

    location_ids = ts_data["pickup_location_id"].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    for location_id in tqdm(location_ids):
        # keep only rides for this 'location_id' (filter operation)
        ts_data_one_location = ts_data.loc[ts_data.pickup_location_id == location_id,['pickup_hour','rides_count']].sort_values(by="pickup_hour")

        # precompute cutoff indices to split  dataframe rows into sequences
        indices = get_cutoff_indices(ts_data_one_location,input_seq_length,step_size)

        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_length), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []

        # looping through the indices to create features and targets
        for i,idx in enumerate(indices):
            x[i,:] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides_count'].values # features
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides_count'].values # target
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # numpy array to pandas dataframe
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_length))],
        )
        
        features_one_location["pickup_hour"] = pickup_hours
        features_one_location["pickup_location_id"] = location_id

        targets_one_location = pd.DataFrame(
            y,
            columns=["target_rides_next_hour"],
        )

        # concatenate features and targets
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])
    
    features.reset_index(drop=True,inplace=True)
    targets.reset_index(drop=True,inplace=True)
        
    return features, targets["target_rides_next_hour"]
