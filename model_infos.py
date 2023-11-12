
params_model = {
    's2p': {
        'window_size': 599, # needs to be odd
        'crop':None, # default is None, could use small subset to try to overfit the model for debugging
        'header':0,
        'num_appliances': 1,
        'batch_size': 1000,
        'lr': 1e-3,
        'optimizer':'adam',
        'criterion':'BCEWithLogitsLoss',
        'num_epochs': 10,
        'printfreq': 100,
        'n_dense': 1,
        'transfer_cnn': False
    },
    'TransformerSeq2Seq': {
        'window_size': 21, # doesn'thave to be odd
        'num_appliances': 1,
        'crop': None, # default is None, use one batch to try to overfit the model for debugging
        'header':0,
        'batch_size': 32,
        'lr': 1e-3,
        'optimizer':'adam',
        'criterion':'BCEWithLogitsLoss',
        'num_epochs':100,
        'printfreq': 1,
        'd_model': 512,
        'n_head': 8,
        'num_encoder_layers': 6
    },
    'TransformerSeq2Point': {
        'window_size': 21, # needs to be odd
        'num_appliances': 1,
        'crop': None, # default is None, could use small subset to try to overfit the model for debugging
        'header':0,
        'batch_size': 32,
        'lr': 1e-3,
        'optimizer':'adam',
        'criterion':'BCEWithLogitsLoss',
        'num_epochs':100,
        'printfreq': 1,
        'd_model': 512,
        'n_head': 8,
        'num_encoder_layers': 6
    },
    'attention_cnn_Pytorch':{
        'window_size': 599, # needs to be odd
        'crop':None, # default is None, could use small subset to try to overfit the model for debugging
        'header':0,
        'num_appliances': 1,
        'batch_size': 1000,
        'lr': 1e-2,
        'optimizer':'adam',
        'criterion':'BCEWithLogitsLoss',
        'num_epochs': 10,
        'printfreq': 100,
        'n_dense': 1,
        'transfer_cnn': False
    }
}


columns_names = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
houses = {
    '1': [10, 10, 1, 10, 5],
    '2': [8, 5, 1, 10, 2],
    '3': [9, 8, 2, 10, 6],
    '4': [9, 8, 1, 4, 6],  # 4 and 6 washingmachines
    '5': [8, 7, 1, 4, 3],
    '6': [7, 6, 1, 3, 2],
    '7': [9, 10, 1, 6, 5],
    '8': [9, 8, 1, 10, 4],
    '9': [7, 6, 1, 4, 3],
    '10': [10, 8, 4, 6, 5],
    '12': [6, 5, 1, 10, 10],
    #'13': [9, 8, 4, 6, 5],
    '15': [10, 7, 1, 4, 3],
    '16': [10, 10, 1, 6, 5],
    '17': [8, 7, 2, 10, 4],
    '18': [10, 9, 1, 6, 5],
    '19': [5, 4, 1, 10, 2],
    '20': [9, 8, 1, 5, 4],
}