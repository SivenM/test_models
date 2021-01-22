import pandas as pd
import numpy as np
import os


class TresholdFinder:

    def __init__(self, data_path, save_path):
        self.main_dir = data_path
        self.save_path = save_path
        self.dir_model_names = os.listdir(self.main_dir)
        self.main()

    def calc_acc(self, tp, fn, fp, tn):
        return (tp + tn) / (tp + fp + fn + tn)

    def _get_csv_names(self, dir_path):
        return os.listdir(dir_path)

    def read_csv(self, csv_path):
        return pd.read_csv(csv_path)

    def get_tp_fn(self, human_df, treshold):
        tp = human_df['max_conf'][human_df['max_conf'] > treshold].to_numpy().shape[0]
        fn = human_df['max_conf'][human_df['max_conf'] < treshold].to_numpy().shape[0]
        return tp, fn

    def get_fp_tn(self, bg_df, treshold):
        fp = bg_df['max_conf'][bg_df['max_conf'] > treshold].to_numpy().shape[0]
        tn = bg_df['max_conf'][bg_df['max_conf'] < treshold].to_numpy().shape[0]
        return fp, tn

    def calculate_treshold(self, bg_df, human_df):
        some_list = []
        acc_list = []
        treshold_list = []
        treshold = 0.01
        for i in range(100):
            if treshold > 1:
                best_treshold = 1
                break
            tp, fn = self.get_tp_fn(human_df, treshold)
            fp, tn = self.get_fp_tn(bg_df, treshold)
            some_list.append([tp, fn, fp, tn])
            acc = self.calc_acc(tp, fn, fp, tn)
            #print(acc)
            acc_list.append(acc)
            treshold_list.append(treshold)
            treshold += 0.01
        best_acc = max(acc_list)
        best_treshold = treshold_list[acc_list.index(max(acc_list))]
        print(some_list[acc_list.index(max(acc_list))])
        return best_treshold, best_acc
            
    def main(self):
        result_df = pd.DataFrame(columns= ['model_name', 'treshold', 'accuracy'])
        for i, dir_model_name in enumerate(self.dir_model_names):
            print(f'    Произвожу расчет порога модели {dir_model_name}^')
            csv_dir_path = os.path.join(self.main_dir, dir_model_name)
            csv_names = self._get_csv_names(csv_dir_path)
            csv_bg_path = os.path.join(csv_dir_path, csv_names[0])   
            csv_human_path = os.path.join(csv_dir_path, csv_names[1])   
            print('     Загружаю данные...')
            bg_df = self.read_csv(csv_bg_path)
            human_df = self.read_csv(csv_human_path)
            print('     Произвожу расчет и оценку...')
            best_treshold, best_acc = self.calculate_treshold(bg_df, human_df)
            print('     Сохранение результатов')
            result_df.loc[i] = [dir_model_name, best_treshold, best_acc]
        result_df.to_csv(self.save_path)


if __name__ == "__main__":
    data_path = 'testing_files\\data\\csv_output_data'
    save_path = 'testing_files\\data\\result\\result4.csv'
    TresholdFinder(data_path, save_path)