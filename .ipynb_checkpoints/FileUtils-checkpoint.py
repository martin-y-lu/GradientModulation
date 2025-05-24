import os
import pickle
def load_log_data(file_name):
    # print(os.getcwd(),os.listdir())
    with open(file_name, 'rb') as file: 
        # Call load method to deserialze 
        log_data = pickle.load(file) 
    return log_data
def write_log_data( log_data,file_name = 'log_data_comparing_activations.pkl'):
    if os.path.exists(file_name):
        print("file already exists")
    else:
        with open(file_name, 'wb') as file: 
            # A new file will be created 
            pickle.dump(log_data, file) 