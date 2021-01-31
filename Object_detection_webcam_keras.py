import argparse
parser = argparse.ArgumentParser(description='Python Mask Detection')
parser.add_argument('-r', '--remote', action='store_true', help='Using RDP Webcam')
parser.add_argument('-s', '--stream', action='store_true', help='Using iPhone camera stream over rtsp as video input')
args = parser.parse_args()

from multiprocessing import Process
import subprocess
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def stream():
    info('function stream')
    with open(os.devnull, 'wb') as devnull:
        subprocess.check_call(['sudo', 'modprobe', 'v4l2loopback', 'devices=1', 'exclusive_caps=1'], stdout=devnull, stderr=subprocess.STDOUT)
        subprocess.check_call(['ffmpeg', '-i', 'rtsp://@192.168.1.217:554/', '-vcodec', 'rawvideo', '-f', 'v4l2', '/dev/video0'], stdout=devnull, stderr=subprocess.STDOUT)
    # os.system('sudo modprobe v4l2loopback devices=1 exclusive_caps=1')
    # os.system(f'ffmpeg -i rtsp://@192.168.1.217:554/ -vcodec rawvideo -f v4l2 /dev/video0')

######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a webcam feed.
# It draws boxes, scores, and labels around the objects of interest in each frame
# from the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

def detection():
    # Import packages
    import os
    import cv2
    import numpy as np
    import tensorflow as tf
    import sys
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    #from tensorflow.keras.models import load_model

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")

    # Import utilites
    from utils import label_map_util
    from utils import visualization_utils as vis_util

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    # PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # # Path to label map file
    # PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 2

    ## Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    # detection_graph = tf.Graph()
    # with detection_graph.as_default():
    #     od_graph_def = tf.compat.v1.GraphDef()
    #     with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    #         serialized_graph = fid.read()
    #         od_graph_def.ParseFromString(serialized_graph)
    #         tf.import_graph_def(od_graph_def, name='')

    #     tf.io.write_graph(detection_graph, "./export", "mask_detection.pb", False)
    #     sess = tf.compat.v1.Session(graph=detection_graph)


    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # # Output tensors are the detection boxes, scores, and classes
    # # Each box represents a part of the image where a particular object was detected
    # detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # # Each score represents level of confidence for each of the objects.
    # # The score is shown on the result image, together with the class label.
    # detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # # Number of objects detected
    # num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = "/home/pi/Desktop/deploy.prototxt.txt"
    weightsPath = "/home/pi/Desktop/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = tf.keras.models.load_model("/home/pi/Desktop/mask_model.h5")

    def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                try:
                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 150x150, and preprocess it
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)

                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))
                except:
                    pass

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            def process_img(x):
                return x

            imgs = [process_img(i) for i in faces]
            # imgs = np.concatenate(*imgs, axis=0) if len(faces) > 1 else imgs
            preds = maskNet.predict(np.vstack(faces), batch_size=len(faces))

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    # Initialize webcam feed
    video = cv2.VideoCapture(1 if args.remote else 0)
    ret = video.set(3,1280)
    ret = video.set(4,720)

    while(True):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        """# Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.80)
        """

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # cv2_imshow(frame)
        height, width, layers = frame.shape
        size = (width,height)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    stream = Process(target=stream, args=())
    if args.stream:
        stream.start()
    detection = Process(target=detection, args=())
    detection.start()
    detection.join()
    try:
        if stream.is_alive():
            stream.close()
    except:
        pass
