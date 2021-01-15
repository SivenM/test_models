
from model import SSD
from write_preds import *

"""
Записвыает предикты моделей из тестового набора данных в jsonы
"""

model_path_list = [
    'testing_files\\data\\models\\model_32_bg2500.h5',
    'testing_files\\data\\models\\model_32_bg5000.h5',
    'testing_files\\data\\models\\model_32_bg8000.h5',
    'testing_files\\data\\models\\model_32_bg10000.h5',
    'testing_files\\data\\models\\model_32_bg12000.h5',
    'testing_files\\data\\models\\model_32_ep500_bg_2500.h5',
    'testing_files\\data\\models\\model_32_ep500_bg_5000.h5',
    'testing_files\\data\\models\\model_32_ep500_bg_8000.h5',
    'testing_files\\data\\models\\model_32_ep500_bg_10000.h5',
    'testing_files\\data\\models\\model_32_ep500_bg_12000.h5',
    ]

path_data_dict = {'tp': ['testing_files\\data\\train_images_32',
                          'testing_files\\data\\train_labels_32.csv'
                          ],
                   'bg': ['testing_files\\data\\test_fn_527',
                          'testing_files\\data\\test_fn_527_ann.csv'
                          ],
                 } 
json_dir = 'testing_files\\data\\json_test_data'        

predict_data(json_dir, path_data_dict, model_path_list)