import tensorflow as tf
import numpy as np
import cv2

PATH_TO_CKPT = 'frozen_inference_graph.pb'
TRAFFIC_LIGHT_CLASS_ID = 10

sess_od = tf.Session()
detection_graph_od = tf.Graph()
with detection_graph_od.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

with detection_graph_od.as_default():
  with tf.Session(graph=detection_graph_od) as sess_od:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph_od.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph_od.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph_od.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph_od.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph_od.get_tensor_by_name('num_detections:0')
                  
def detect_light(path_to_image):
    box = None
    im_height = -1
    im_width = -1
    
    with detection_graph_od.as_default():    
        with tf.Session(graph=detection_graph_od) as sess_od:
            image_np = cv2.imread(path_to_image, 3)
            im_height, im_width, _ = image_np.shape
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess_od.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            if TRAFFIC_LIGHT_CLASS_ID in classes:
                classes_list = classes.flatten().tolist()
                ind_best_tl = classes_list.index(TRAFFIC_LIGHT_CLASS_ID)
                box = boxes[0][ind_best_tl]
                
    if box is not None:
        ymin, xmin, ymax, xmax = box
        box = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    return box

