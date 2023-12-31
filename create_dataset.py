# This file implements NILM datasets
# The file should be formatted as follows:
# date,Aggregate,Appliance1,Apliance2,Appliance3,Appliance4,......
# 2021-10-01 00:05:00,7263.77,5693.27,1376.74,1067.92,4612.52,......
# 2021-10-01 00:10:00,7263.44,5635.79,1338.87,1069.43,4669.22,......
# 2021-10-01 00:15:00,7301.17,5605.89,1323.73,1064.68,4597.48,......

# Token for github: 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

import pandas as pd
import time
import os
import re
import argparse

DATA_DIRECTORY = '../REFIT/New_Data/'
SAVE_PATH = '../REFIT/kettle/'
AGG_MEAN = 522
AGG_STD = 814

TASK = 'classification' # or "Regression"

def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning \
                                     example for NILM')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                          help='The directory containing the CLEAN REFIT data')
    parser.add_argument('--appliance_name', type=str, default='kettle',
                          help='which appliance you want to train: kettle,\
                          microwave,fridge,dishwasher,washingmachine')
    parser.add_argument('--aggregate_mean',type=int,default=AGG_MEAN,
                        help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std',type=int,default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the training data')
    parser.add_argument('--task', type=str, default=TASK,
                          help='Classification or Regression')
    return parser.parse_args()


params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 19, 20],
        'channels': [8, 9, 9, 8, 7, 9, 9, 7, 6, 9, 5, 9],
        'test_house': 2,
        'validation_house': 5,
        'test_on_train_house': 5,
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [4, 10, 12, 17, 19],
        'channels': [8, 8, 3, 7, 4],
        'test_house': 4,
        'validation_house': 17,
        'test_on_train_house': 10,
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [2, 5, 9, 12, 15],
        'channels': [1, 1, 1,  1, 1],
        'test_house': 15,
        'validation_house': 12,
        'test_on_train_house': 5,
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [5, 7, 9, 13, 16, 18, 20],
        'channels': [4, 6, 4, 4, 6, 6, 5],
        'test_house': 20,
        'validation_house': 18,
        'test_on_train_house': 13,
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [2, 5, 7, 8, 9, 15, 16, 17, 18],
        'channels': [2, 3, 5, 4, 3, 3, 5, 4, 5],
        'test_house': 8,
        'validation_house': 18,
        'test_on_train_house': 5,
    }
}


def load(path, building, appliance, channel):

    # load csv
    file_name = path + 'CLEAN_House' + str(building) + '.csv'
    single_csv = pd.read_csv(file_name,
                             header=0,
                             names=['aggregate', appliance],
                             usecols=[1, channel+1], # unix is removed
                             na_filter=False,
                             parse_dates=True,
                             infer_datetime_format=True,
                             memory_map=True
                             )

    return single_csv


def main():
    start_time = time.time()        
    # test path
    # path = '../../../data/refit/CLEAN_REFIT_081116/'
    # save_path = 'refitdata/'
    
    args = get_arguments()
    
    appliance_name = args.appliance_name
    print(appliance_name)
    
    path = args.data_dir
    save_path = args.save_path
    save_path = '../REFIT/' + appliance_name + '/' + TASK + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(path)
    aggregate_mean = args.aggregate_mean#522
    aggregate_std = args.aggregate_std#814  
    
    total_length = 0
    print("Starting creating dataset...")
    # Looking for proper files
    for idx, filename in enumerate(os.listdir(path)):
        if os.path.isfile(os.path.join(path, filename)):
            single_step_time = time.time()

            # test
            if filename == 'CLEAN_House' + str(params_appliance[appliance_name]['test_house']) + '.csv':
                print('File: ' + filename + ' test set')
                # Loading
                test = load(path,
                    params_appliance[appliance_name]['test_house'],
                    appliance_name,
                    params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses']
                            .index(params_appliance[appliance_name]['test_house'])]
                    )
        
                # Normalization
                test['aggregate'] = (test['aggregate'] - aggregate_mean) / aggregate_std
                if TASK == 'classification':
                    test[appliance_name] = (test[appliance_name] > params_appliance[appliance_name]['on_power_threshold']).astype(int)
                else:
                    test[appliance_name] = \
                        (test[appliance_name] - params_appliance[appliance_name]['mean']) / params_appliance[appliance_name]['std']
        
                # Save
                test.to_csv(save_path + appliance_name + '_test_' + 'H' + str(params_appliance[appliance_name]['test_house'])
                            + '.csv', index=False)
        
                print("Size of test set is {:.3f} M rows (House {:d})."
                    .format(test.shape[0] / 10 ** 6, params_appliance[appliance_name]['test_house']))
                del test
        
            # val
            elif filename == 'CLEAN_House' + str(params_appliance[appliance_name]['validation_house']) + '.csv':
                print('File: ' + filename + ' validation set')
                # Loading
                val = load(path,
                    params_appliance[appliance_name]['validation_house'],
                    appliance_name,
                    params_appliance[appliance_name]['channels']
                    [params_appliance[appliance_name]['houses']
                            .index(params_appliance[appliance_name]['validation_house'])]
                    )
        
                # Normalization
                val['aggregate'] = (val['aggregate'] - aggregate_mean) / aggregate_std
                if TASK == 'classification':
                    val[appliance_name] = (val[appliance_name] > params_appliance[appliance_name]['on_power_threshold']).astype(int)
                else:
                    val[appliance_name] = \
                        (val[appliance_name] - params_appliance[appliance_name]['mean']) / params_appliance[appliance_name]['std']
        
                # Save
                val.to_csv(save_path + appliance_name + '_validation_' + 'H' + str(params_appliance[appliance_name]['validation_house'])
                        + '.csv', index=False)
        
                print("Size of validation set is {:.3f} M rows (House {:d})."
                    .format(val.shape[0] / 10 ** 6, params_appliance[appliance_name]['validation_house']))
                del val
            
            elif (re.search(r'\d+', filename) == None):
                print(f"No house number found in filename: {filename}")
            
            elif int(re.search(r'\d+', filename).group()) in params_appliance[appliance_name]['houses']:
                print('File: ' + filename)
                print('    House: ' + re.search(r'\d+', filename).group())
        
                # Loading
                try:
                    csv = load(path,
                            int(re.search(r'\d+', filename).group()),
                            appliance_name,
                            params_appliance[appliance_name]['channels']
                            [params_appliance[appliance_name]['houses']
                                    .index(int(re.search(r'\d+', filename).group()))]
                            )
        
                    # Normalization
                    csv['aggregate'] = (csv['aggregate'] - aggregate_mean) / aggregate_std
                    if TASK == 'classification':
                        csv[appliance_name] = (csv[appliance_name] > params_appliance[appliance_name]['on_power_threshold']).astype(int)
                    else:
                        csv[appliance_name] = \
                            (csv[appliance_name] - params_appliance[appliance_name]['mean']) / params_appliance[appliance_name][
                                'std']
            
                    rows, columns = csv.shape
                    total_length += rows
        
                    if filename == 'CLEAN_House' + str(params_appliance[appliance_name]['test_on_train_house']) + '.csv':
                        csv.to_csv(save_path + appliance_name + '_test_on_train_' + 'H' + str(
                            params_appliance[appliance_name]['test_on_train_house'])
                                + '.csv', index=False)
                        print("Size of test on train set is {:.3f} M rows (House {:d})."
                            .format(csv.shape[0] / 10 ** 6, params_appliance[appliance_name]['test_on_train_house']))
        
                    # saving the whole merged file
                    csv.to_csv(save_path + appliance_name + '_training_.csv', mode='a', index=False, header=False)
        
                    del csv
        
                except:
                    pass
        
                #print('    total_partial length: {}'.format(total_length / 10 ** 6))
        else:
            print(f"Skipping directory: {filename}")

    print("Size of training set is {:.3f} M rows.".format(total_length / 10 ** 6))
    print("\nNormalization parameters: ")
    print("Mean and standard deviation values USED for AGGREGATE are:")
    print("    Mean = {:d}, STD = {:d}".format(aggregate_mean, aggregate_std))
    print('Mean and standard deviation values USED for ' + appliance_name + ' are:')
    print("    Mean = {:d}, STD = {:d}"
          .format(params_appliance[appliance_name]['mean'], params_appliance[appliance_name]['std']))
    print("\nTraining, validation and test sets are  in: " + save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
    
if __name__ == '__main__':
    main()


    