import os
import numpy as np
import utils


# тестовое чтение данных

test_json_path = 'testing_files\\data\\json_test_data\\model_32_bg2500\\bg\\model_32_bg2500.h5_tp_2.jpg.json'
data = utils.get_json_data(test_json_path)
print(data['gt_true'])


