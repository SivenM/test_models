from utils import *
import tensorflow as tf
import numpy as np


class LabelEncoder:
    """Трансформирует аннотации в таргеты для обучения.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.
    Этот класс имеет операции для создания таргетов для пакета сэмплов, 
    которые состоят из входных изображений, ограничивающих рамок для 
    присутствующих объектов и их идентификаторов классов.

    Attributes:
      anchor_box: генератор анкер боксов.
      box_variance: Коэффициенты масштабирования, используемые для масштабирования
        целевых объектов ограничивающей рамки
    """

    def __init__(self):
        self._anchor_box = create_prior_boxes()

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.4, ignore_iou=0.4
    ):
        """Сопоставляет gt боксы с анкербоксами через IOU.

        1. вычисляет попарную IOU для M `anchor_boxes` and N `gt_boxes`
           и выдает `(M, N)` размером матрицу.
        2. gt боксы с максимальным IOU в каждой строке назначается анкер бокс при 
           при условии что IOU больше чем `match_iou`.
        3. Если максимальная IOU в строке меньше чем `ignore_iou`, анкер боксу 
           назначается фоновый класс
        4. Остальные блоки, которым не назначен класс игнорируются

        Arguments:
          anchor_boxes: Тензор размором `(total_anchors, 4)`
            представляет все анкербоксыrepresenting для данной формы входного 
            изображения, где каждый анкер бокс форматом `[x, y, width, height]`.
          gt_boxes: Ттензор размером `(num_objects, 4)` представляющие gt боксы,
           где формат бокса`[x, y, width, height]`.
          match_iou: Значение представляющее минимальный порог IOU для определения
           того, может ли gt бокс назначен анкер боксу
          ignore_iou: Значение представляющее максимальный порог IOU для определения
            анкер боксу класс фона

        Returns:
          matched_gt_idx: Индес найденного объекта
          positive_mask: маска анкер боксов, которым назначены gt боксы.
          ignore_mask: маска анкер боксов, которая игнорируется во времяобучения
        """
        iou_matrix = compute_iou1(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(iou_matrix, match_iou)
        negative_mask = tf.less(iou_matrix, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Трансформирует gt боксы в таргеты для обучения"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        #box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, gt_boxes, cls_ids):
        """Создает боксы и классифициет таргеты для одиночного сэмпла"""
        anchor_boxes = self._anchor_box
        gt_boxes = tf.cast(gt_boxes, dtype=tf.float32)
        gt_boxes = tf.reshape(gt_boxes, ((1,) + gt_boxes.shape))
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        gt_boxes = convert_to_xywh(gt_boxes)
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, gt_boxes)

        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), 0, cls_ids
        )
        #cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_bg = tf.cast(tf.equal(cls_target, 0.), dtype=tf.float32)
        #print(cls_target)
        #cls_target = tf.expand_dims(cls_target, axis=-1)
        
        label = tf.concat([box_target, cls_target], axis=-1)
        label = tf.concat([label, cls_bg], axis=-1)

        return label

    def encode_batch(self, gt_boxes, cls_ids):
        """Создает боксы и классифицирует таргеты для батча"""
        images_shape = tf.shape(gt_boxes)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        return labels.stack()

    def encode_bg(self, images_num):
        batch_size = images_num 

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = tf.zeros([84, 4], dtype=tf.float32)
            human_cls = tf.zeros([84, 1], dtype=tf.float32)
            bg_cls = tf.ones([84, 1], dtype=tf.float32)
            label = tf.concat([label, human_cls], axis=-1)
            label = tf.concat([label, bg_cls], axis=-1)
            labels = labels.write(i, label)
        return labels.stack()


class EncodeData:

    def __init__(self, image_path, labels_path=None):
        self.image_path = image_path
        self.labels_path = labels_path
        self.image_name_list = self._get_image_names()
        self.labels_df = self._read_labels()
        self.label_encoder = LabelEncoder()

    def _get_image_names(self):
        return os.listdir(self.image_path)

    def _read_labels(self):
        if self.labels_path != None:
            return pd.read_csv(self.labels_path)
        else:
            return 0

    def _encode(self):
        x = []
        y = []
        names_df = self.labels_df['filename']
        for image_name in self.image_name_list:
            image_path = self.image_path + '/' + image_name
            image = tf.keras.preprocessing.image.load_img(image_path,
                                                          color_mode = "grayscale",
                                                          target_size=(32, 32))
            image = tf.keras.preprocessing.image.img_to_array(image)
            index_bbox = names_df.index[names_df == image_name]
            bbox_coords = self.labels_df.iloc[index_bbox[0], 4:]
            bbox_coords = [bbox_coords['xmin'],              
                           bbox_coords['ymin'],
                           bbox_coords['xmax'],
                           bbox_coords['ymax']] 
            x.append(image)
            y.append(bbox_coords)
        return tf.convert_to_tensor(x), tf.convert_to_tensor(y, dtype=tf.float32)

    def create_tp_data(self):
        X, Y = self._encode() 
        X_train = X / 255       
        cls = tf.ones((Y.shape[0], 1), dtype=tf.float32)
        Y_train = self.label_encoder.encode_batch(Y, cls)
        return X_train, Y_train, 

    def create_bg_data(self):
        X = self._encode() 
        X_train = X / 255
        img_names = self.image_name_list
        return X_train, img_names  

    def create_test_bg_dataset(self):
        """
        Создает тестовые false negative данные
        """
        X, Y, img_names = self.create_bg_data()
        img_numbers = X.shape[0]
        return (X, Y, img_names)

    def create_test_people_dataset(self):
        """
        Создает тестовые true positive данные
        """
        X, Y = self._encode() 
        X = X / 255       
        cls = tf.ones((Y.shape[0], 1), dtype=tf.float32)
        treshold = int(X.shape[0] * 0.1)
        treshold = X.shape[0] - treshold
        #X_train = X[:treshold]
        X_test = X[treshold:]
        Y_test = Y[treshold:]
        img_names = self.image_name_list[treshold:]
        return (X_test, Y_test, img_names) 


class JsonWriter:
    """
    Записывает данные предсказаний модели
    """
    def __init__(self, json_dir):
        self.json_dir = json_dir

    def write_json(self, save_path, data):
        with open(save_path, "w", encoding="utf8") as write_file:
            json.dump(data, write_file, ensure_ascii=False) 

    def create_data(self, model_name, img_name, boxes, cls_predictions, gt_true):
        return {
                "model_name": model_name,
                "img_name": img_name,
                "boxes": boxes, 
                "cls_predictions": cls_predictions, 
                "gt_true": gt_true}

    def write(self,
              key,
              model_name,
              img_name,
              boxes,
              cls_predictions,
              gt_true,
              ):
        """
        Пишет данные модели и ее предсказания в json файл.
        """
        save_path = self.json_dir + '/' + model_name + '_' + 'tp' + '_' + img_name + '.json'
        data = self.create_data(model_name, img_name, boxes, cls_predictions, gt_true)
        self.write_json(save_path, data)