
from shapely.geometry import Point, Polygon
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
   
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    vid = cv2.VideoCapture(video_path)

    out = None

    frame_num = 0
    arr = np.array([[1 , 2 , 3 , 4, 5, 6],]) 
    arr_cnts = np.array([[1 , 2 , 3, 4, 5],])
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        coords1 = [(0, 260), (570, 305), (570, 835), (407, 877), (355,704), (0,681)]
        zone1 = Polygon(coords1)
        coords2 = [(677, 870), (661, 370), (1022, 362), (1073, 634), (757,874)]
        zone2 = Polygon(coords2)
        coords3 = [(1056, 655), (1075, 1076), (1920, 1080), (1920, 606), (1417,540)]
        zone3 = Polygon(coords3)
        coords4 = [(357, 846), (370, 709), (0, 730), (0, 911)]
        zone4 = Polygon(coords4)
        zone1_counter = 0
        zone2_counter = 0
        zone3_counter = 0
        zone4_counter = 0


        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
       
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]      

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr() # only used with visualisation
           
        # auto enabled counting of object in video
            x_cen = (int(bbox[0])+int(bbox[2]))/2
            y_cen = (int(bbox[1])+int(bbox[3]))/2
            f_width = int(bbox[2]) - int(bbox[0])
            f_height = int(bbox[3]) - int(bbox[1])
            point = Point(x_cen, y_cen)
            if zone1.contains(point):
                zone1_counter+=1
            if zone2.contains(point):
                zone2_counter+=1
            if zone3.contains(point):
                zone3_counter+=1
            if zone4.contains(point):
                zone4_counter+=1
            arr = np.append(arr , [np.array([frame_num, int(track.track_id), x_cen, y_cen , f_width , f_height])], axis=0 )

        # collect the data per frame
        arr_cnts = np.append(arr_cnts , [np.array([frame_num, zone1_counter, zone2_counter, zone3_counter, zone4_counter])], axis=0 )

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
    
    # Calculate the video duration

    feed_fps = vid.get(cv2.CAP_PROP_FPS)
    totalNoFrames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    durationInSeconds = float(totalNoFrames) / float(feed_fps)
    print(f'Total video duration was {durationInSeconds}')
    print(f'Total number of frame was {totalNoFrames}')

    # Tracks array to pd.df and timestamp calculations
    arr = np.delete(arr,0,0)
    res_df = pd.DataFrame(arr, columns = ['frame_no', 'track_id', 'x', 'y', 'width', 'height'])
    # TO DO: extract from vid
    vid_start = pd.Timestamp('2017-01-01 12:00:00')
    vid_lenght = 10
    frame_max = res_df['frame_no'].max()
    time_per_frame = int(vid_lenght)/int(frame_max)
    def time_flow(frame_index):
        return vid_start + pd.to_timedelta(((int(frame_index) - 1)*time_per_frame), unit='s')  
    res_df['timestamp'] = res_df['frame_no'].apply(time_flow)
    res_df['frame_no'] = res_df['frame_no'].astype('int')
    res_df['track_id'] = res_df['track_id'].astype('int')
    res_df.to_csv("./outputs/tracks.csv",index=False)
    # Counts array to pd.df and timestamp calculations
    arr_cnts = np.delete(arr_cnts,0,0)
    counts_df = pd.DataFrame(arr_cnts, columns = ['frame_no', 'zone1_count', 'zone2_count', 'zone3_count', 'zone4_count'])
    counts_df['timestamp'] = counts_df['frame_no'].apply(time_flow)
    counts_df.to_csv("./outputs/counts.csv",index=False)
    print(f'max frame from df {frame_max}')
    print(res_df.tail())
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

