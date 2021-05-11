import numpy as np
import tensorflow as tf
import os
import tensorflow_model_optimization as tfmot

class AncorBoxCreator:
    """
    Генерирует анкербоксы
    """
    def __init__(self, fmap_dims={'det_2': 1}, obj_scales={'det_2': 0.9}, aspect_ratios={'det_2': [1.]}, img_size=64):
        self.img_size = img_size
        self.fmap_dims = fmap_dims             
        self.obj_scales = obj_scales 
        self.aspect_ratios = aspect_ratios
        self.fmaps = list(self.fmap_dims.keys())

    def create_boxes(self):
        prior_boxes = []    
        for k, fmap in enumerate(self.fmaps):
          for i in range(self.fmap_dims[fmap]):
            for j in range(self.fmap_dims[fmap]):
              cx = (j + 0.5) / self.fmap_dims[fmap]
              cy = (i + 0.5) / self.fmap_dims[fmap]  
              for ratio in self.aspect_ratios[fmap]:
                prior_boxes.append([cx, cy, self.obj_scales[fmap] * np.sqrt(ratio), self.obj_scales[fmap] / np.sqrt(ratio)])
        prior_boxes = tf.convert_to_tensor(prior_boxes) 
        return prior_boxes * self.img_size


class SSD:

    def __init__(self, model_path, img_size):
        self.model_path = model_path
        self.model = self.load_model()
        self.anchor_boxes = AncorBoxCreator(img_size=img_size).create_boxes()

    def load_model(self):
        return tf.keras.models.load_model(self.model_path)

    def load_img(self, img_path):
        image = tf.keras.preprocessing.image.load_img(img_path,
                                                        color_mode = "grayscale",
                                                        target_size=(32, 32))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array /= 255
        image_array = np.reshape(image_array, ((1,) + image_array.shape))
        return image_array

    def convert_to_corners(self, boxes):
        return tf.concat(
            [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
            axis=-1,)

    def create_prior_boxes(self):
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
        #prior_boxes = tf.clip_by_value(prior_boxes, 0, 1)

        return prior_boxes * 32

    def create_prior_boxes2(self):
        fmap_dims = {'det_2': 1}

        obj_scales = {
                      'det_2': 0.9,
                      } 
        aspect_ratios = {
                          'det_2': [1., 2., 0.5],
                          } 
        fmaps = list(fmap_dims.keys())  
        prior_boxes = []    
        for k, fmap in enumerate(fmaps):
          for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
              cx = (j + 0.5) / fmap_dims[fmap]
              cy = (i + 0.5) / fmap_dims[fmap]  
              for ratio in aspect_ratios[fmap]:
                prior_boxes.append([cx, cy, obj_scales[fmap] * np.sqrt(ratio), obj_scales[fmap] / np.sqrt(ratio)])
        prior_boxes = tf.convert_to_tensor(prior_boxes) 
        prior_boxes = tf.clip_by_value(prior_boxes, 0, 1)
        return prior_boxes * 64

    def ssd_prediction(self, feature_maps, num_classes):                                                
        batch_size = feature_maps[0].shape[0] 
        predicted_features_list = []
        for feature in feature_maps: 
            predicted_features_list.append(tf.reshape(tensor=feature, shape=(batch_size, -1, num_classes + 4))) #изменяем форму массива с картами признаков одного размера
        predicted_features = tf.concat(values=predicted_features_list, axis=1)
        return predicted_features

    def decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = self.convert_to_corners(boxes)
        return boxes_transformed

    def get_boxes_cls(self, img_array, num_prior_boxes=84):
        y_pred = self.model(img_array)
        y_pred = self.ssd_prediction(feature_maps=y_pred, num_classes=2)
        if num_prior_boxes == 3:
            anchor_boxes = self.create_prior_boxes2()
        else:
            anchor_boxes = self.create_prior_boxes()
        box_predictions = y_pred[:, :, :4]
        cls_predictions = tf.nn.softmax(y_pred[:, :, 4:])
        boxes = self.decode_box_predictions(anchor_boxes[None, ...], box_predictions)
        return boxes, cls_predictions

    def get_human_box_idx(self, cls):
        idx_cls = tf.argmax(cls[0], axis=1)
        human_box_idx = []
        for i, idx in enumerate(idx_cls.numpy()):
            if idx == 0:
                human_box_idx.append(i)
        return human_box_idx

    def get_human_boxes(self, boxes, human_box_idx):
        human_box_list = []
        for i in human_box_idx:
            human_box_list.append(boxes.numpy()[0][i])
        return human_box_list

    def predict(self, img_path):
        img_array = self.load_img(img_path)
        boxes, cls_predictions = self.get_boxes_cls(img_array)
        human_box_idx = self.get_human_box_idx(cls_predictions)
        human_box_list = self.get_human_boxes(boxes, human_box_idx)
        return img_array, human_box_list

    def compute_iou(self, boxes, gt_true):
        """
        Считает IOU между gt_true И предсказанными боксами
        """
        lu = np.maximum(boxes[:, :2], gt_true[:2])
        rd = np.minimum(boxes[:, 2:], gt_true[2:])
        intersection = np.maximum(0.0, rd - lu)
        intersection_area = intersection[:, 0] * intersection[:, 1]
        box1_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box2_area = (gt_true[2] - gt_true[0]) * (gt_true[3] - gt_true[1])
        union_area = np.maximum(box1_area + box2_area - intersection_area, 1e-8)
        return np.clip(intersection_area / union_area, 0.0, 1.0)
    
    def test(self, img_array):
        image_array = tf.reshape(img_array, ((1,) + img_array.shape))
        boxes, cls_predictions = self.get_boxes_cls(image_array)
        return boxes, cls_predictions 
        
    def test2(self, img_array):
        image_array = tf.reshape(img_array, ((1,) + img_array.shape))
        boxes, cls_predictions = self.get_boxes_cls(image_array)
        return boxes, cls_predictions 
    
    def evaluate(self, img_array):
        image_array = tf.reshape(img_array, ((1,) + img_array.shape))
        boxes, cls_predictions = self.get_boxes_cls(image_array)
        idx_cls = tf.argmax(cls_predictions[0], axis=1)
        human_box_idx = []
        for i, idx in enumerate(idx_cls.numpy()):
            if idx == 0:
                if cls_predictions[0, i, 0] == 1:
                    human_box_idx.append(i)
        if len(human_box_idx) == 0:
            conf = 0



class Model2head:

    def __init__(self, model_path, img_size=64, q=False):
        self.model_path = model_path
        self.img_size = img_size
        self.q = q
        self.model = self.load_model() 
        
    def load_model(self):
        if self.q:
            return tf.lite.Interpreter(model_path=self.model_path)
        else:
            return tf.keras.models.load_model(self.model_path)

    def test(self, img_array):
        image_array = tf.reshape(img_array, ((1,) + img_array.shape))
        if self.q:
            self.model.allocate_tensors()

            input_index = self.model.get_input_details()[0]["index"]
            output_box_index = self.model.get_output_details()[0]["index"]
            output_conf_index = self.model.get_output_details()[1]["index"]

            self.model.set_tensor(input_index, image_array)
            self.model.invoke()

            output_box = self.model.tensor(output_box_index)
            output_conf = self.model.tensor(output_conf_index)

            cls_val = tf.argmax(output_conf()[0], axis=0).numpy().tolist()
            if cls_val == 0:
                box = output_box()[0].tolist()
            elif cls_val == 1:
                box = []
        else:
            y_pred = self.model(image_array, training=False)
            cls_val = tf.argmax(y_pred[1], axis=-1).numpy().tolist()[0]
            if cls_val == 0:
                box = y_pred[0][0].numpy().tolist()
            elif cls_val == 1:
                box = []
        return box, cls_val 
        