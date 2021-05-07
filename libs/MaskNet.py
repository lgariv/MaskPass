import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from time import sleep, time
import threading
from pydub import AudioSegment
from pydub.playback import play

print("[INFO] loading face detector model...")
prototxtPath = os.path.expanduser("/home/pi/CollegeProject/models/FaceDetector/deploy.prototxt.txt")
weightsPath = os.path.expanduser(
    "/home/pi/CollegeProject/models/FaceDetector/res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(
    # model_path=os.path.expanduser("/home/pi/Desktop/mask_model.tflite"))
    model_path=os.path.expanduser("/home/pi/CollegeProject/models/MaskDetector/model.tflite"))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def detect_faces(frame):
    '''returns cropped image and bounding box'''
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (200, 200),
                                    (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    locs = []
    faces = []

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
                # extract the face ROI
                # startX, startY: top left corner
                # endX, endY: bottom right corner
                face = frame[startY:endY, startX:endX]

                # resize it to 160x160
                face = cv2.resize(face, (160, 160))

                # expand array shape from [1, 160, 160, 3] to [160, 160, 3]
                face = np.expand_dims(face, axis=0)

                # add the face and bounding boxes to their respective lists
                faces.append(face)
                locs.append((startY, startX, endY, endX))
            except:
                pass
    return locs, faces


def predict(faces):
    '''tflite'''
    labels = []
    scores = []
    for face in faces:
        # pre-process image to conform to MobileNetV2
        input_data = preprocess_input(np.float32(face))

        # set our input tensor to our face image
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # perform classification
        interpreter.invoke()

        # get our output results tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        result = np.squeeze(output_data)

        # get label from the result.
        # the class with the higher confidence is the label.
        (mask, withoutMask) = result
        label = "Mask" if mask > withoutMask else "No Mask"

        # get the highest confidence as the label's score
        score = np.max(result)

        labels.append(label)
        scores.append(score)
    return (labels, scores)


def play_no_mask_video():
    video_path = "/home/pi/CollegeProject/mask-instructions.mp4"
    audio_path = "/home/pi/CollegeProject/mask-instructions.aac"
    audio = AudioSegment.from_file(audio_path)

    def play_audio(audio):
        '''
        it takes opencv roughly a third of
        a second to start playing the video,
        so we delay the audio in order to
        synchronize it with the video.
        '''
        sleep(1 / 3)
        play(audio)

    t = threading.Thread(
        target=play_audio, args=(audio, )
    )  # we used threading in order to play the audio and video simultanously.
    t.start()

    frame_rate = 30
    time_before_frame = time()  # reset time
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            time_elapsed = time() - time_before_frame
            if time_elapsed < 1. / frame_rate:  # making sure the playback frame rate isn't faster than it should be.
                sleep((1. / frame_rate) - time_elapsed)  # if it is too fast, it will wait for the remaining time before loading the next frame.
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow('Mask Detection', frame)
            time_before_frame = time()
            cv2.waitKey(1)
        else:
            break
    cap.release()


def play_high_temp_screen():
    high_temp_frame = cv2.imread("/home/pi/CollegeProject/images/too-hot.jpg")
    high_temp_frame = cv2.resize(high_temp_frame, (640, 480))
    cv2.imshow('Mask Detection', high_temp_frame)
    cv2.waitKey(5000)


def play_entry_allowed_screen():
    entry_allowed_frame = cv2.imread("/home/pi/CollegeProject/images/OK.jpg")
    entry_allowed_frame = cv2.resize(entry_allowed_frame, (640, 480))
    cv2.imshow('Mask Detection', entry_allowed_frame)
    cv2.waitKey(5000)


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    """https://www.digikey.com/en/maker/projects/how-to-perform-object-detection-with-tensorflow-lite-on-raspberry-pi/b929e1519c7c43d5b2c6f89984883588"""
    def __init__(self, resolution=(640, 480)):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC,
                              cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while (1):
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


def draw_boxes_with_predictions(frame, locs, pred_labels, scores, temps):
    for (box, pred_label, score, temp) in zip(locs, pred_labels, scores, temps):
        # unpack the bounding box and predictions
        (startY, startX, endY, endX) = box
        # determine the class label and color we'll use to draw the bounding box and text
        label = "{}, {:.2f}c".format(pred_label, temp)
        color = (0, 0, 255) if "No Mask" in label or temp >= 38 else (0, 255, 0)
        # include the probability (as percentages) in the label
        label = "{}: {:.2f}%".format(label, float(score) * 100)
        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    return frame
