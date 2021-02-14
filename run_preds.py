
from model import SSD
from write_preds import *

"""
Записвыает предикты моделей из тестового набора данных в jsonы
"""

model_path_list = [
    'testing_files\\ratio_data2\\models\\m32_ratio1_k0.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio1_k1.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio1_k2.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio1_k3.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio1_k4.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio2_k0.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio2_k1.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio2_k2.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio2_k3.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio2_k4.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio3_k0.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio3_k1.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio3_k2.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio3_k3.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio3_k4.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio4_k0.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio4_k1.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio4_k2.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio4_k3.h5',
    'testing_files\\ratio_data2\\models\\m32_ratio4_k4.h5',
    ]

model_path = ['testing_files\\ratio_data3\\models\\m32_ratio1_k0.h5']

path_data_dict = {
                    'tp': [
                         'testing_files\\data\\train_images_32',
                         'testing_files\\data\\train_labels_32.csv'
                        ],
                   'bg': [
                          'testing_files\\data\\test_fn_527',
                          'testing_files\\data\\test_fn_527_ann.csv'
                         ],
                 } 
json_dir = 'testing_files\\ratio_data3\\json_preds_data'        
predict_data(json_dir, path_data_dict, model_path)