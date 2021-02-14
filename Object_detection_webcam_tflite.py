import argparse
parser = argparse.ArgumentParser(description='Python Mask Detection')
parser.add_argument('-r',
                    '--remote',
                    action='store_true',
                    help='Using RDP Webcam')
parser.add_argument('-s',
                    '--stream',
                    action='store_true',
                    help='Using iPhone camera stream over rtsp as video input')
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
        subprocess.check_call([
            'sudo', 'modprobe', 'v4l2loopback', 'devices=1', 'exclusive_caps=1'
        ],
                              stdout=devnull,
                              stderr=subprocess.STDOUT)
        subprocess.check_call([
            'ffmpeg', '-i', 'rtsp://@192.168.1.217:554/', '-vcodec',
            'rawvideo', '-f', 'v4l2', '/dev/video0'
        ],
                              stdout=devnull,
                              stderr=subprocess.STDOUT)
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
import numpy as np
import cv2
import tensorflow as tf

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
interpreter = tf.lite.Interpreter(
    model_path=os.path.expanduser("~/Desktop/mask_model.tflite"))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def infer(i, img):
    input_mean = input_std = float(127.5)
    input_data = (np.float32(img) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    result = np.squeeze(output_data)
    (mask, withoutMask) = result
    label = 1 if mask > withoutMask else 2
    class_label = int(label)
    score = np.max(result)
    return class_label, score


def detection():
    # Import packages
    import os
    import sys
    from tensorflow.keras.preprocessing.image import img_to_array

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")

    # Import utilites
    from models.research.object_detection.utils import label_map_util
    from models.research.object_detection.utils import visualization_utils as vis_util

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
    prototxtPath = "~/Desktop/deploy.prototxt.txt"
    weightsPath = "~/Desktop/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    def detect_and_predict_mask(frame, faceNet):
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
        classes = []
        scores = []

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

                left = float(startX)
                top = float(startY)
                right = float(endX)
                bottom = float(endY)

                # left = left - (left*0.1)
                # top = top - (top*0.1)
                # right = right + (right*0.1)
                # bottom = bottom + (bottom*0.1)

                (startX, startY, endX, endY) = (int(left), int(top),
                                                int(right), int(bottom))

                try:
                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 150x150, and preprocess it
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    # face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)

                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    width, height, _ = frame.shape
                    locs.append((startY / width, startX / height, endY / width,
                                 endX / height))
                except:
                    pass

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            # preds = []

            import multiprocessing
            from multiprocessing import Pool
            # print(f'CPU number: {multiprocessing.cpu_count()}')
            with Pool(processes=multiprocessing.cpu_count()) as pool:
                preds = pool.starmap(infer, enumerate(faces))
                (classes, scores) = preds[0]
                classes = np.array([classes])
                scores = np.array([scores])

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, classes, scores)

    # Initialize webcam feed
    video = cv2.VideoCapture(1 if args.remote else 0)
    ret = video.set(3, 1280)
    ret = video.set(4, 720)

    while (True):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)
        """
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        """

        (locs, classes, scores) = detect_and_predict_mask(frame, faceNet)

        # print(np.array(locs), np.array(classes).astype(np.int32), np.array(scores))
        category_index = {
            1: {
                'id': 1,
                'name': 'With Mask'
            },
            2: {
                'id': 2,
                'name': 'Without Mask'
            }
        }
        # """
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.array(locs),
            np.array(classes).astype(np.int32),
            np.array(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.80)
        # """

        # loop over the detected face locations and their corresponding
        # locations
        # for (box, pred) in zip(locs, preds):
        #     # unpack the bounding box and predictions
        #     (startX, startY, endX, endY) = box
        #     (mask, withoutMask) = pred

        #     # determine the class label and color we'll use to draw
        #     # the bounding box and text
        #     label = "Mask" if mask > withoutMask else "No Mask"
        #     color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        #     # include the probability in the label
        #     label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        #     # display the label and bounding box rectangle on the output
        #     # frame
        #     cv2.putText(frame, label, (startX, startY - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        #     cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # cv2_imshow(frame)
        height, width, layers = frame.shape
        size = (width, height)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
        cv2.setWindowProperty('ObjectDetector', cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

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
