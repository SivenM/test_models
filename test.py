import os 
import argparse
import json


path = 'testing_files\\ratio_data2\\models'
model_names = os.listdir(path)
models_path = [os.path.join(path, model_name) for model_name in model_names]
#print(models_path)

def get_models_path(config):
    model_names = os.listdir(confg['model_path'])
    models_path = [os.path.join(path, model_name) for model_name in model_names]
    return models_path


def get_path_data_dict(config):
    path_data_dict = {
                    'tp': [
                         config['tp_img'],
                         config['tp_ann']
                        ],
                   'bg': [
                          config['bg_img'],
                          config['bg_nn']
                         ],
                 }
    return path_data_dict


def get_output_dir(confg):
    return confg['output_dir']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file', dest='config', type=argparse.FileType('r'), default=None, help='cfg file in json format')
    args = parser.parse_args()
    if args.config:
        config = json.load(args.config)
        models_path = get_models_path
        path_data_dict = get_path_data_dict(config)
        output_dir = get_output_dir