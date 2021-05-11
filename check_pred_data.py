"""
Программа находит боксы с IoU > 0.75 и среди них берет max conf предикта.
Результаты сохранятются в csv-файлы

"""


import os
import numpy as np
import data_preprocesing
import pandas as pd
import argparse
import json
#import utils

# тестовое чтение данных
#test_json_path = 'testing_files\\data\\json_test_data\\model_32_bg2500\\bg\\model_32_bg2500.h5_tp_2.jpg.json'
#data = utils.get_json_data(test_json_path)
#print(data['gt_true'])
######################################


class ConfDataWriter:
    
    def __init__(self, main_test_json_path, output_path, model_type):
        self.model_type = model_type
        self.main_test_json_path = main_test_json_path
        self.output_path = output_path
        self.tmaster = data_preprocesing.TrecholdMaster()
        self.jreader = data_preprocesing.JsonReader(main_test_json_path)
        #self.jwriter = data_preprocesing.JsonWriter()  
        self.main()       

    def create_df(self):
        return pd.DataFrame(columns=['img_name', 'max_conf', 'iou', 'box'])

    def save_df(self, df, save_path):
        df.to_csv(save_path)

    def parse_data(self, data):
        boxes = np.array(data['boxes'])[0]
        cls_conf = np.array(data["cls_predictions"])[0]
        gt_true = np.array(data["gt_true"])
        img_name = data['img_name']
        return boxes, cls_conf, gt_true, img_name

    def parse_data_2head(self, data):
        boxes = np.array(data['boxes'])
        cls_conf = np.array(data["cls_predictions"])
        gt_true = np.array(data["gt_true"])
        img_name = data['img_name']
        return boxes, cls_conf, gt_true, img_name

    def get_human_conf(self, model_name, dir_model_data, save_dir):
        human_confs = self.create_df()
        type_data = 'tp'
        print(f'Тип данных: {type_data}')
        jsons_dir = os.path.join(dir_model_data, type_data)
        json_names_list = os.listdir(jsons_dir)
        print(f'Количество изображений: {len(json_names_list)}')
        for i, json_name in enumerate(json_names_list):
            json_path = os.path.join(jsons_dir, json_name)
            data = self.jreader.get_json_data(json_path)
            if self.model_type == '2head':
                box, cls_val, gt_true, img_name = self.parse_data_2head(data)
                if box.shape[0] == 0:
                    iou = 0
                else:
                    iou = self.tmaster.compute_iou(np.expand_dims(box, axis=0), gt_true)[0]
                if cls_val == 0:
                    human_confs.loc[i] = [img_name, 1, iou, box.tolist()]
                elif cls_val == 1:
                    human_confs.loc[i] = [img_name, 0, iou, box.tolist()]
            else:
                conf, iou, box = self.tmaster.get_human_conf_box_iou(boxes, cls_conf, gt_true)                
                human_confs.loc[i] = [img_name, conf, iou, box]
        csv_save_name = 'output' + '_' + model_name + '_' + type_data + '.csv'
        csv_save_path = os.path.join(save_dir, csv_save_name)
        self.save_df(human_confs, csv_save_path)


    def get_bg_conf(self, model_name, dir_model_data, save_dir):
        bg_confs = self.create_df()
        type_data = 'bg'
        print(f'Тип данных: {type_data}')
        jsons_dir = os.path.join(dir_model_data, type_data)
        json_names_list = os.listdir(jsons_dir)
        number_json_names = len(json_names_list)
        print(f'Количество изображений: {number_json_names}')
        for i, json_name in enumerate(json_names_list):
            json_path = os.path.join(jsons_dir, json_name)
            data = self.jreader.get_json_data(json_path)
            if self.model_type == '2head':
                box, cls_val, gt_true, img_name = self.parse_data_2head(data)
                if cls_val == 0:
                    bg_confs.loc[i] = [img_name, 1, '-', box.tolist()]
                elif cls_val == 1:
                    bg_confs.loc[i] = [img_name, 0, '-', box.tolist()]
            else:
                boxes, cls_conf, gt_true, img_name = self.parse_data(data)
                max_conf_idx = np.argmax(cls_conf[:, 0])
                max_conf = cls_conf[max_conf_idx, 0]
                if max_conf != 1.:
                    max_conf = 0
                max_box = boxes[max_conf_idx]
                bg_confs.loc[i] = [img_name, max_conf, '-', max_box.tolist()]
        csv_save_name = 'output' + '_' + model_name + '_' + type_data + '.csv'
        csv_save_path = os.path.join(save_dir, csv_save_name)
        self.save_df(bg_confs, csv_save_path)    

    def main(self):
        os.mkdir(self.output_path)
        for model_name in self.jreader.dir_name_list:
            dir_model_data = os.path.join(main_test_json_path, model_name)
            #human_confs = {'tp': {'img_name': [],
            #                      'conf': [],
            #                      'box': []},
            #                'tn': [],
            #                'bg': {'img_name': [],
            #                      'conf': [],
            #                      'box': []}}
            print(f'Предикты модели {model_name}')
            save_dir = os.path.join(self.output_path, model_name)
            os.mkdir(save_dir) 
            self.get_human_conf(model_name, dir_model_data, save_dir)
            self.get_bg_conf(model_name, dir_model_data, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file', dest='config', type=argparse.FileType('r'), default=None, help='cfg file in json format')
    args = parser.parse_args()
    if args.config:
        config = json.load(args.config)
        main_test_json_path = config['json_pred_dir']
        output_path = config['csv_conf_data']
        model_type = config['model_type']
    ConfDataWriter(main_test_json_path, output_path, model_type)