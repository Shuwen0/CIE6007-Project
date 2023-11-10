import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import os

HOUSE_IDX = 1
debug = 1
storage = 1
AGG_MEAN = 522
AGG_STD = 814
params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [1, 2],
        'channels': [10, 8],
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [1, 2],
        'channels': [13, 15],
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [1, 2],
        'channels': [12, 14],
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [1, 2],
        'channels': [6, 13],
    },
    'washing_machine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [1, 2],
        'channels': [5, 12],
    }
}
appliance_list = list(params_appliance.keys())

def load_dataframe(house, appliance, col_names=['time', 'data'], nrows=None):
    file_name = os.path.join(house, appliance)
    df = pd.read_table(file_name +'.dat',
                       sep="\s+",
                       nrows=nrows,
                       usecols=[0, 1],
                       names=col_names,
                       dtype={'time': str},
                       )
    return df

def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning \
                                     example for NILM')
    parser.add_argument('--house', type=str, default=HOUSE_IDX,
                          help='The target house idx')
    parser.add_argument('--resample_seconds', type=int, default=8, help='The desired resampling seconds')
    return parser.parse_args()


args = get_arguments()
house = args.house
sample_seconds = args.resample_seconds
print('House ',house)


def main():
    house_dir = "House_" + str(house)

    mains_df = load_dataframe(house_dir, 'aggregate')
    washing_machine_df = load_dataframe(house_dir, 'washing_machine', ['time', 'washing_machine'])
    kettle_df = load_dataframe(house_dir, 'kettle', ['time', 'kettle'])
    microwave_df = load_dataframe(house_dir, 'microwave', ['time', 'microwave'])
    fridge_df = load_dataframe(house_dir, 'fridge', ['time', 'fridge'])
    dishwasher_df = load_dataframe(house_dir, 'dishwasher', ['time', 'dishwasher'])

    mains_df['time'] = pd.to_datetime(mains_df['time'], unit='s')
    mains_df.set_index('time', inplace=True)
    mains_df.columns = ['aggregate']
    mains_df.reset_index(inplace=True)

    # Ensure that all appliance DataFrames have 'time' as datetime objects
    appliances_dfs = [washing_machine_df, kettle_df, microwave_df, fridge_df, dishwasher_df]
    for df in appliances_dfs:
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

    # mains conversion
    mains_df.set_index('time', inplace=True)

    # Concatenate all DataFrames along the column axis
    print("Starting concatenation......")
    all_dfs = [mains_df] + appliances_dfs  # Add mains_df to the list of appliance DataFrames
    combined_df = pd.concat(all_dfs, axis=1, join='outer')
    print("Successfully concatenated!")

    # Resample and fill missing values if needed
    print("Starting resampling......")
    combined_df_resampled = combined_df.resample(str(sample_seconds) + 'S').mean().fillna(method='backfill')
    combined_df_resampled = combined_df_resampled.dropna()
    print("Successfully resampled to %d seconds" % sample_seconds)

    if debug:

        # If 'time' is the index and is already in datetime format
        timestamp_start = combined_df_resampled.index.min().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_end = combined_df_resampled.index.max().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Timestamp range is from {timestamp_start} to {timestamp_end}")

        # Ensure the index is sorted before inferring frequency
        combined_df_resampled = combined_df_resampled.sort_index()
        frequency = combined_df_resampled.index.inferred_freq
        print(f"The inferred frequency of the DataFrame is {frequency}")

    # Reset index
    combined_df_resampled.reset_index(inplace=True)

    # Normzalization
    combined_df_resampled['aggregate'] = (combined_df_resampled['aggregate'] - AGG_MEAN) / AGG_STD

    # THresholding
    for appliance in appliance_list:
        threshold = params_appliance[appliance]['on_power_threshold']
        print('maximum value of ' + appliance + ' is ' + str(combined_df_resampled[appliance].max()))
        combined_df_resampled[appliance] = (combined_df_resampled[appliance] > threshold).astype(int)


    # storage
    if storage:
        new_filename = os.path.join('..','New_Data', 'House_' + str(house) + '.csv')
        base_path = os.getcwd()  # or os.path.dirname(__file__) in a .py script
        full_path = os.path.join(base_path, '..', 'New_Data')
        # Check if the directory exists; if not, create it
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        combined_df_resampled.to_csv(new_filename, index=False)

main()

