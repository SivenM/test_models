import os
import argparse
import json
from model import SSD
from write_preds import *

"""
Записвыает предикты моделей из тестового набора данных в jsonы
"""

def get_models_path(config):
    model_names = os.listdir(config['model_path'])
    models_path = [os.path.join(config['model_path'], model_name) for model_name in model_names]
    return models_path


def get_path_data_dict(config):
    path_data_dict = {
                    'tp': [
                         config['tp_img'],
                         config['tp_ann']
                        ],
                    'bg': [
                          config['bg_img'],
                          config['bg_ann']
                         ],
                 }
    return path_data_dict


def get_output_dir(confg):
    return confg['json_pred_dir']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file', dest='config', type=argparse.FileType('r'), default=None, help='cfg file in json format')
    args = parser.parse_args()
    if args.config:
        config = json.load(args.config)
        models_path = get_models_path(config)
        path_data_dict = get_path_data_dict(config)
        output_dir = get_output_dir(config)
        img_size = config['img_size']
        model_type = config['model_type']
        quantized = config['quantized']
    predict_data(output_dir, path_data_dict, models_path, img_size, model_type, quantized)