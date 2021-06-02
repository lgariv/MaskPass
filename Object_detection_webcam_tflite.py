#!/usr/bin/env python3

from time import sleep
import cv2
from threading import Thread
from libs.MaskNet import VideoStream, detect_faces, predict, draw_boxes_with_predictions, play_no_mask_video, play_high_temp_screen, play_entry_allowed_screen
from libs.MLX90640 import get_scaled_temp_image, temp_average_from_bbox
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(13, GPIO.OUT)  # GPIO13 (pin 33)
servo = GPIO.PWM(13, 50)  # GPIO13 (pin 33) for servo, operating at 50Hz
servo.start(12)
sleep(0.5)
servo.ChangeDutyCycle(0)


def detection():
    authorized_frames_count = 0
    non_authorized_frames_count = 0

    # Initialize webcam feed
    camera = VideoStream().start()

    while (1):
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = camera.grabbed, camera.read()
        if not ret:
            input(
                "No camera device found. Please plug a USB webcam to the Raspberry Pi.\nPress any key to continue..."
            )
            continue  # skips the rest of the commands for the current loop

        # use the face detector model to extract bounding boxes and pre-processed faces from the frame
        locs, faces = detect_faces(frame)

        # run mask prediction using our mask classification model and retrieve predicted label and confidence
        pred_labels, scores = predict(faces)

        temps = []
        thermal_image = get_scaled_temp_image()
        for bbox in locs:
            temp = temp_average_from_bbox(thermal_image, bbox)
            temps.append(temp)

        # loop over the detected face locations and their corresponding locations
        frame = draw_boxes_with_predictions(frame, locs, pred_labels, scores,
                                            temps)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Mask Detection', frame)
        cv2.moveWindow('Mask Detection', 0, 0)
        cv2.setWindowProperty('Mask Detection', cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        if not pred_labels:  # check if list is empty
            authorized_frames_count = 0
            non_authorized_frames_count = 0
        elif ("No Mask" in pred_labels) or (max(temps) >= 38):
            authorized_frames_count = 0
            non_authorized_frames_count += 1
        else:
            non_authorized_frames_count = 0
            authorized_frames_count += 1

        if non_authorized_frames_count == 5:
            if "No Mask" in pred_labels:
                camera.stop()
                non_authorized_frames_count = 0
                authorized_frames_count = 0
                play_no_mask_video()
                camera = VideoStream().start()
            else:
                camera.stop()
                non_authorized_frames_count = 0
                authorized_frames_count = 0
                play_high_temp_screen()
                camera = VideoStream().start()
        elif authorized_frames_count == 5:
            ''' code to allow opening the door '''
            """
            # servo used is SG90:
            # datasheet specifies that at 50Hz the range of 0 to 180 degrees
            # can be targeted at 0.4ms to 2.4ms (or 2% to 12% duty cycle),
            # where 0.4ms (2% dc) is 0 degrees, 1.4ms (7% dc) is 90 degrees,
            # and 2.4ms (12% dc) is 180 degrees.
            """
            camera.stop()
            screen = Thread(target=play_entry_allowed_screen, args=())
            screen.start()

            # Turn servo to 90 degrees
            servo.ChangeDutyCycle(7)
            sleep(0.5)
            servo.ChangeDutyCycle(0)

            # Wait for 2 seconds
            sleep(2)

            # Turn servo back to 0 degrees
            servo.ChangeDutyCycle(12)
            sleep(0.5)
            servo.ChangeDutyCycle(0)

            # Wait for 2 seconds
            sleep(2)
            camera = VideoStream().start()

        # Press 'q' to break out of the loop and quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    # video.release()
    camera.stop()
    cv2.destroyAllWindows()


detection()