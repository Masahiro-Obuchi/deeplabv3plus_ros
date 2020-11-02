#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import rosparam

# from PIL import Image

import tensorflow as tf

import time

rospy.init_node('deeplabv3_plus', anonymous=True)

class SegmentImage(object):
    def __init__(self):
        self._image_pub = rospy.Publisher(
            '/seg_result', Image, queue_size=1)
        self._seg_map_pub = rospy.Publisher(
          '/seg_map', Image, queue_size=1)
        self._image_sub = rospy.Subscriber(
            '/image', Image, self.image_callback, queue_size=1, buff_size=5000000)
        self._bridge = CvBridge()

    def image_callback(self, data):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError, e:
            print e

        try:
            seg_img,seg_map = self.segmentation(cv_image)
            pub_img = self._bridge.cv2_to_imgmsg(seg_img)
            pub_map = self._bridge.cv2_to_imgmsg(seg_map)
            pub_img.header.frame_id = data.header.frame_id
            pub_img.header.stamp = data.header.stamp
            pub_map.header.frame_id = data.header.frame_id
            pub_map.header.stamp = data.header.stamp
            self._image_pub.publish(pub_img)
            self._seg_map_pub.publish(pub_map)
        except CvBridgeError, e:
            print e

    def segmentation(self, img):
        segmented_image, segmented_map = run_visualization(img)
        return segmented_image, segmented_map


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, frozen_graph_filename):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        resized_image = image
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map):
    seg_image = label_to_color_image(seg_map).astype(np.uint8)

    result = cv2.add(image, seg_image)
    return result, seg_image


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])


FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

model_path = rospy.get_param('~model_path')

MODEL = DeepLabModel(model_path)

def run_visualization(img):
    height, width = img.shape[:2]
    size = (width, height)
    original_img = cv2.resize(img, (480, 320))
    resized_img, seg_map = MODEL.run(original_img)
    seg_map = np.array(seg_map, dtype='uint8')
    resized_img = cv2.resize(resized_img, size, interpolation=cv2.INTER_LINEAR)
    seg_map = cv2.resize(seg_map, size, interpolation=cv2.INTER_LINEAR) # resize
    seg_img, seg_map_img = vis_segmentation(resized_img, seg_map)
    return seg_img, seg_map_img


if __name__ == '__main__':
    si = SegmentImage()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
