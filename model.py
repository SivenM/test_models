import numpy as np
import tensorflow as tf

class SSD:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

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

    def get_boxes_cls(self, img_array):
        y_pred = self.model(img_array)
        y_pred = self.ssd_prediction(feature_maps=y_pred, num_classes=2)
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

    def test(self, img_array):
        boxes, cls_predictions = self.get_boxes_cls(img_array)
        return boxes, cls_predictions 