from data_preprocesing import EncodeData, JsonWriter
from model import SSD


def create_dataset(path_data_dict):
    """
    Создает тестовый датасет для проверки моделей.

    path_data_dict: словарь, значения которого содержат пути к данным.
    test_size: скаляр, который означает размер изображения.

    return:
        data: Словарь. Хранит матрицы данных. Значение словаря - это корртеж,
        который хранит матрицу изображения, аннотацию и имя изображения
    """
    data = {}
    for key, data_path in path_data_dict:
        encode_data = EncodeData(data_path[0], data_path[1])        
        if key == 'tp':
            tp = encode_data.create_test_people_dataset()
            data['tp'] = tp
        elif key == 'bg':
            bg = encode_data.create_test_bg_dataset()
            data['bg'] = bg
    return data

def main(json_dir, path_data_dict, model_path_list):
    
    """
    Основной цикл программы.

    1. Создается тестовый датасет
    2. Берется модель из списка
    3. Модель прогонятеся по тестовому датасету
    4. Результаты записываются в .jsonы
    5. цикл повторяется с 2ого по 4ый пункт пока не прогонятся все модели.
    """

    test_datasets = create_dataset(path_data_dict)
    
    for model_path in model_path_list:
        model_name = model_path.split('/')[-1]
        json_path = json_dir + '/' + model_name.split('.')[0]
        os.mkdir(json_path)
        json_writer = JsonWriter(json_path)
        model = SSD(model_path)
        for key, dataset in test_datasets:
            x_test = dataset[0]
            y_test = dataset[1]
            img_names = dataset[2]
            for i in range(x_test.shape[0]):
                img_array = x_test[i]
                gt_true = y_test[i]
                img_name = img_names[i]

                boxes, cls_predictions = model.test(img_array)
                json_writer.write(
                    key,
                    model_name,
                    img_name,
                    boxes.numpy().tolist(),
                    cls_predictions.numpy().tolist(),
                    gt_true,
                )
    


if __name__ == "__main__":
    path_data_dict = {'tp': ['F:\\ieos\\data\\train_images\\train_images_32',
                             'F:\\ieos\\data\\train_images\\train_labels_32.csv'
                             ],
                      'bg': ['F:\\ieos\\data\\train_images\\set_test_fn_imgs',
                             'F:\\ieos\\data\\train_images\\test_fn_527_ann.csv'
                             ],
                      }       