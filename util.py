import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimate_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    z = np.zeros(len(__data_columns))
    z[0] = sqft
    z[1] = bath
    z[2] = bhk
    if loc_index >= 0:
        z[loc_index] = 1

    return round(__model.predict([z])[0],2)

def get_location_names():
    return __locations
def get_data_columns():
    return __data_columns
def load_saved_artifacts():
    print("loading saved artifacts...starts")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json",'r')as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    if __model is None:
        with open("./artifacts/banglore_home_prices_model.pickle",'rb') as f:
            __model = pickle.load(f)
    print("Loading done")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimate_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimate_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimate_price('Kalhalli', 1000, 2, 2))
    print(get_estimate_price('Ejipura', 1000, 2, 2))
