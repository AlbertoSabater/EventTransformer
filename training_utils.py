import os
import datetime


def create_folder_if_not_exists(folder_path):
    if not os.path.isdir(folder_path): os.makedirs(folder_path)

def get_model_name(model_dir):
    folder_num = len(os.listdir(model_dir))
    return '/{}_model_{}/'.format(datetime.datetime.today().strftime('%m%d_%H%M'), folder_num)

def create_model_folder(path_results, folder_name):
    
    path_model = path_results + folder_name
    
    # Create logs and model main folder
    create_folder_if_not_exists(path_model)

    model_name = get_model_name(path_results + folder_name)
    path_model += model_name
    create_folder_if_not_exists(path_model)
    create_folder_if_not_exists(path_model + 'weights/')
    
    return path_model


def update_params(original_params, new_params):
    for k,v in new_params.items():
        # print(k,v)
        if type(v) == dict and k in original_params: v = update_params(original_params[k], new_params[k])
        original_params[k] = v
    return original_params



