## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import os
import tensorflow as tf

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
##config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

MODEL_NAME = 'new_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')
NUM_CLASSES = 1
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
# print('Model loaded')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print('Label map loaded')

print('Detection task started...')
PATH_TO_TEST_IMAGES_DIR = 'images'
TEST_IMAGE_PATHS = ['/home/gaurav/Object recognition/models/research/object_detection/images/img_000045.png']

# Size, in inches, of the output images.
IMAGE_SIZE = (24, 16)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ##print('inside while')
        depth_image = np.asanyarray(depth_frame.get_data())
        ##color_image = np.asanyarray(color_frame.get_data())
        global depth_colormap
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        ##images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', depth_colormap)
        cv2.waitKey(1)



        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:

                image_np = depth_colormap
                image_np_expanded = np.expand_dims(depth_colormap, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                cv2.imshow('object detection', cv2.resize(depth_colormap, (800, 600)))
                cv2.waitKey(1)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()


finally:

    pipeline.stop()
