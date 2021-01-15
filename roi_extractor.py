import tensorflow as tf
import os, sys
import requests as http
import json
import datetime
from random import random
from time import sleep 
    
# import horovod.tensorflow as hvd
# hvd.init()
os.environ['CUDA_VISIBLE_DEVICES'] = "3" #str(hvd.local_rank())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 3GB memory each
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])

    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    # print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
      pass
    # Virtual devices must be set before GPUs have been initialized



from object_detection import DetectObject
from scipy.spatial.distance import euclidean
from tools import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import preprocessing, nn_matching
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
import numpy as np
import cv2 
from PIL import Image
from core.functions import *
from core.config import cfg
from tensorflow.python.saved_model import tag_constants
from core.yolov4 import filter_boxes
import core.utils as utils
from absl.flags import FLAGS
from absl import app, flags, logging
import time
import os


# define constants
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov3', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to input video or set to 0 for webcam')
flags.DEFINE_string('json_export', TIMESTAMP_DIR+'pickled_json.json',
                    'path to json data')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.70, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', True, 'count objects being tracked on screen')
flags.DEFINE_boolean('crop', True, 'crop detections from images')

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


minimapArray = []
def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    interpreter = None

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(
            FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    filename = video_path.split('.')[-2]
    VideoOut = None
    MinimapOut = None

    
    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        time_milli = vid.get(cv2.CAP_PROP_POS_MSEC)
        time_milli = time_milli/1000

        # set frame per seconds
        vid.set(cv2.CAP_PROP_FPS, 1000)
        env.FPS = 1000
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        VideoOut = cv2.VideoWriter(TIMESTAMP_DIR +'objectDetector.avi', codec, fps, (width, height))
        
        


    frame_num = 0
    count = 10
    ObjectDetector = DetectObject()

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            # print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1

        # pass in the object detector
        ObjectDetector.interpreter = interpreter
        bboxes, frame, result = ObjectDetector.analyzeDetection(return_value, frame, frame_num, FLAGS,
        infer, encoder, nms_max_overlap, tracker)
        
        # loop through the bounding box and export into the ROI folder.
        for i, j in bboxes.items():
            xmin, ymin, w, h = int(j[0]), int(j[1]), int(j[2]), int(j[3])
            if w <= 0 or h <= 0:
                pass
            else:
                # ROI Extraction
                maskedImage = frame[ymin:ymin+h, xmin:xmin+w]

        
                roi_name= "./ROI/ROI_frame_%s.jpg" %(str(round(time.time() * 1000)))
                cv2.imwrite(roi_name, maskedImage) # save transformed image to path

    

if __name__ == '__main__':
    try:
        app.run(main)

    except SystemExit:
        pass
