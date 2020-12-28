from data_preprocesing import EncodeData, JsonWriter
from model import SSD


def create_dataset(path_data_dict, test_size):
    data = []
    for key, data_path in data_path_dict:
        encode_data = EncodeData(data_path[0], data_path[1])        
        if key == 'tp':
            tp = encode_data.create_test_people_dataset()
            data.append(tp)
        elif key == 'bg':
            bg = encode_data.create_test_bg_dataset(test_size)
            data.append(bg)
    return data

def main(json_dir, path_data_dict, model_path_list, test_size):
    test_datasets = create_dataset(path_data_dict, test_size)
    
    for model_path in model_path_list:
        model_name = model_path.split('/')[-1]
        json_path = json_dir + '/' + model_name.split('.')[0]
        json_writer = JsonWriter(json_path)
        model = SSD(model_path)
        for dataset in test_datasets:
            x_test = dataset[0]
            y_test = dataset[1]
            img_names = dataset[2]
            for i in range(x_test.shape[0]):
                img_array = x_test[i]
                gt_true = y_test[i]
                img_name = img_names[i]

                boxes, conf = model.test(img_array)
                json_writer.write
    
