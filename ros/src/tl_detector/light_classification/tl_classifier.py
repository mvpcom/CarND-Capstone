from styx_msgs.msg import TrafficLight
from keras.models import load_model
import tensorflow as tef
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        self.model = load_model('light_classification/model.h5')
        self.model._make_predict_function()
        self.graph = tef.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            input_image (cv::Mat): image containing the traffic light
        Returns:
            tl_state (int): ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        tl_class = TrafficLight.UNKNOWN
        image = cv2.resize(image, (160, 160), interpolation = cv2.INTER_AREA)
        image = np.expand_dims(image, axis=0)
        with self.graph.as_default():
            pred = round(self.model.predict(image))
        if pred == 0:
            tl_class = TrafficLight.RED
        else:
            tl_class = TrafficLight.GREEN
        return tl_class

