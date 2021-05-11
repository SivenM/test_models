from data_preprocesing import EncodeData, JsonWriter
from model import SSD, Model2head
import os

def create_dataset(path_data_dict, img_size):
    """
    Создает тестовый датасет для проверки моделей.

    path_data_dict: словарь, значения которого содержат пути к данным.
    test_size: скаляр, который означает размер изображения.

    return:
        data: Словарь. Хранит матрицы данных. Значение словаря - это корртеж,
        который хранит матрицу изображения, аннотацию и имя изображения
    """
    data = {}
    if img_size == 32:
        for key, data_path in path_data_dict.items():
            encode_data = EncodeData(data_path[0], data_path[1])        
            if key == 'tp':
                tp = encode_data.create_test_people_dataset()
                data['tp'] = tp
            elif key == 'bg':
                bg = encode_data.create_test_bg_dataset()
                data['bg'] = bg
    elif img_size == 64:
        for key, data_path in path_data_dict.items():
            encode_data = EncodeData(data_path[0], data_path[1])        
            if key == 'tp':
                tp = encode_data.create_test_people_dataset_64()
                data['tp'] = tp
            elif key == 'bg':
                bg = encode_data.create_test_bg_dataset_64()
                data['bg'] = bg
    return data

def predict_data(json_dir, path_data_dict, model_path_list, img_size, model_type, quantized):
    
    """
    Основной цикл программы.

    1. Создается тестовый датасет
    2. Берется модель из списка
    3. Модель прогонятеся по тестовому датасету
    4. Результаты записываются в .jsonы
    5. цикл повторяется с 2ого по 4ый пункт пока не прогонятся все модели.
    """

    print("Загружаю датасет")
    test_datasets = create_dataset(path_data_dict, img_size)
    print("Датасет загружен")
    print("="*80)
    print(f"количество tp изображений: {test_datasets['tp'][0].shape[0]}")
    print(f"количество tp изображений: {test_datasets['bg'][0].shape[0]}")
    os.mkdir(json_dir)
    for model_path in model_path_list:
        model_name = model_path.split('\\')[-1]
        print(f"\nИмя модели: {model_name}")
        print('Прогон тестового набора...')
        json_model_dir_path = os.path.join(json_dir, model_name.split('.')[0])
        os.mkdir(json_model_dir_path)
        if model_type == '2head':
            if quantized == '+':
                q = True
            else:
                q = False

            model = Model2head(model_path, q=q)
        else:
            model = SSD(model_path, img_size)
        for key, dataset in test_datasets.items():
            json_save_path = os.path.join(json_model_dir_path, key)
            os.mkdir(json_save_path)
            json_writer = JsonWriter(json_save_path)
            x_test = dataset[0]
            y_test = dataset[1]
            img_names = dataset[2]
            for i in range(x_test.shape[0]):
                img_array = x_test[i]
                gt_true = y_test[i]
                img_name = img_names[i]
                if model_type == '2head':
                    boxes, cls_predictions = model.test(img_array)
                    json_writer.write(
                        key,
                        model_name,
                        img_name,
                        boxes,
                        cls_predictions,
                        gt_true.numpy().tolist(),
                        )
                else:
                    boxes, cls_predictions = model.test(img_array)
                    json_writer.write(
                        key,
                        model_name,
                        img_name,
                        boxes.numpy().tolist(),
                        cls_predictions.numpy().tolist(),
                        gt_true.numpy().tolist(),
                    )
                #print(f"Результаты прогона тестового изображения {img_name} успешно записаны!")
        
    


#if __name__ == "__main__":
#    path_data_dict = {'tp': ['F:\\ieos\\data\\train_images\\train_images_32',
#                             'F:\\ieos\\data\\train_images\\train_labels_32.csv'
#                             ],
#                      'bg': ['F:\\ieos\\data\\train_images\\set_test_fn_imgs',
#                             'F:\\ieos\\data\\train_images\\test_fn_527_ann.csv'
#                             ],
#                      } 

