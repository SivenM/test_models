import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import json
import cv2


def swap_xy(boxes):
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,)
    

def convert_to_xywh2(box):
    return tf.concat(
        [(box[:2] + box[2:]) / 2.0, box[2:] - box[:2]],
        axis=-1,)


def convert_to_corners(boxes):
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,)


def compute_iou1(boxes1, boxes2):
    """Вычисляет попарно IOU для двух тензоров

    Arguments:
      boxes1: Тензор размером `(N, 4)` представляющий ббоксы,
        где каждый бокс размером `[x, y, width, height]`.
      boxes2: Тензор размером `(M, 4)` представляющий ббоксы,
        где каждый бокс размером `[x, y, width, height]`

    Returns:
      Попарную IOU матрицу размером`(N, M)`, где значение i-ой строки
        j-ого столбца IOU между iым боксом and jым боксом из
        boxes1 and boxes2 соответственно.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = boxes2
    lu = tf.maximum(boxes1_corners[:,None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    box1_area = boxes1[:, 2] * boxes1[:, 3]
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = np.maximum(box1_area[:, None] + box2_area - intersection_area, 1e-8)
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def create_prior_boxes():
    fmap_dims = {'det_1': 4,
                 'det_2': 2,
                 'det_3': 1}

    obj_scales = {'det_1': 0.3,
                  'det_2': 0.6,
                  'det_3': 0.9} 

    aspect_ratios = {'det_1': [1., 2., 0.5],
                      'det_2': [1., 2., 0.5],
                      'det_3': [1., 2., 0.5]}  

    fmaps = list(fmap_dims.keys())  
    prior_boxes = []    
    for k, fmap in enumerate(fmaps):
      for i in range(fmap_dims[fmap]):
        for j in range(fmap_dims[fmap]):
          cx = (j + 0.5) / fmap_dims[fmap]
          cy = (i + 0.5) / fmap_dims[fmap]  
          for ratio in aspect_ratios[fmap]:
            prior_boxes.append([cx, cy, obj_scales[fmap] * np.sqrt(ratio), obj_scales[fmap] / np.sqrt(ratio)])
            if ratio == 1.:
              try:
                additional_scale = np.sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
              except IndexError:
                additional_scale = 1.
              prior_boxes.append([cx, cy, additional_scale, additional_scale]) 
    prior_boxes = tf.convert_to_tensor(prior_boxes) 
    return prior_boxes * 32


def ssd_prediction(feature_maps, num_classes):                                                
    batch_size = feature_maps[0].shape[0] 
    predicted_features_list = []
    for feature in feature_maps: 
        predicted_features_list.append(tf.reshape(tensor=feature, shape=(batch_size, -1, num_classes + 4))) #изменяем форму массива с картами признаков одного размера
    predicted_features = tf.concat(values=predicted_features_list, axis=1)
    return predicted_features

def decode_box_predictions(anchor_boxes, box_predictions):
    boxes = box_predictions
    boxes = tf.concat(
        [
            boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
            tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
        ],
        axis=-1,
    )
    boxes_transformed = convert_to_corners(boxes)
    return boxes_transformed

def get_boxes_cls(model, test_x):
    y_pred = model(test_x)
    y_pred = ssd_prediction(feature_maps=y_pred, num_classes=2)
    anchor_boxes = create_prior_boxes()
    box_predictions = y_pred[:, :, :4]
    cls_predictions = tf.nn.softmax(y_pred[:, :, 4:])
    boxes = decode_box_predictions(anchor_boxes[None, ...], box_predictions)
    return boxes, cls_predictions


def visualise_test_data(img_array, points):
    
    img = tf.concat([img_array, img_array, img_array], axis=2)
    img = img.numpy()
    box2 = list(points.numpy())
    img = cv2.rectangle(img, (int(box2[0]), int(box2[1])),
                           (int(box2[2]), int(box2[3])), 
                            (0, 255, 255), 1)
    plt.imshow(img)
    plt.show()


def visualise_data(img):
    img = tf.concat([img, img, img], axis=2)
    plt.matshow(img)
    plt.show()
    return img

def get_rect_with_nms(img, nms_out):
    img = img.numpy()
    for i in range(10):
        img = cv2.rectangle(img, (int(nms_out[0][0][i][0]), int(nms_out[0][0][i][1])),
                           (int(nms_out[0][0][i][2]), int(nms_out[0][0][i][3])), 
                            (0, 255, 255), 1)

    plt.imshow(img)
    plt.show()


def get_rect(img, human_box_list):
    img = img.numpy()
    for box in human_box_list:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                           (int(box[2]), int(box[3])), 
                            (0, 255, 255), 1)
    plt.imshow(img)
    plt.show()


def predict_with_nms(model, test_img):
    boxes, cls_predictions = get_boxes_cls(model, test_img)
    nms_out = tf.image.combined_non_max_suppression(
                tf.expand_dims(boxes, axis=2),
                cls_predictions,
                10,
                10,
                0.4,
                0.4,
                clip_boxes=False)

    print(nms_out)
    img = visualise_data(test_img[0])
    get_rect_with_nms(img, nms_out)


def get_human_box_idx(cls):
    idx_cls = tf.argmax(cls[0], axis=1)
    human_box_idx = []
    for i, idx in enumerate(idx_cls.numpy()):
        if idx == 0:
            human_box_idx.append(i)
    return human_box_idx


def get_human_boxes(boxes, human_box_idx):
    human_box_list = []
    for i in human_box_idx:
        human_box_list.append(boxes.numpy()[0][i])
    return human_box_list


def print_boxes(human_box_list):
    print('Боксы человека:')
    for human_box in human_box_list:
        print(human_box)

def predict(model, test_img):
    boxes, cls_predictions = get_boxes_cls(model, test_img)
    human_box_idx = get_human_box_idx(cls_predictions)
    img = visualise_data(test_img[0])
    if len(human_box_idx) == 0:
        print('Людей не обнаружено')
    else:
        human_box_list = get_human_boxes(boxes, human_box_idx)
        get_rect(img, human_box_list)
        print_boxes(human_box_list)
    
def evaluate(model, test_img):
    boxes, cls_predictions = get_boxes_cls(model, test_img)
    human_box_idx = get_human_box_idx(cls_predictions)
    human_box_list = get_human_boxes(boxes, human_box_idx)
    return human_box_list

def get_json_data(json_path):
    """
    читает json файл и вызворащает dict
    """

    with open(json_path, 'rb') as read_file:
        ann = json.load(read_file, encoding="utf8")
    return ann


def load_history(history_dir_path, history_form, sizes, folds=5):
    data = {}
    for size in sizes:
        data[str(size)] = {}
        for fold in range(folds):
            json_history = os.path.join(history_dir_path, history_form.format(size, fold))
            data[str(size)][str(fold)] = get_json_data(json_history)
    return data


def load_img(img_path, img_size):
    image = tf.keras.preprocessing.image.load_img(img_path,
                                              color_mode = "grayscale",
                                              target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array /= 255
    return image, img_array