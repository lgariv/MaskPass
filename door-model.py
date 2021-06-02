#
# https://pimylifeup.com/raspberry-pi-distance-sensor/
#
import RPi.GPIO as GPIO
from time import sleep, time

GPIO.setmode(GPIO.BCM)
GPIO.setup(12, GPIO.OUT)
door_servo = GPIO.PWM(12, 50)  # pin 32 for servo, operating at 50Hz
door_servo.start(0)  # start servo pulse with 0% duty cycle (disabled)

trigger_pin = 16
echo_pin = 19

GPIO.setup(trigger_pin, GPIO.OUT)  # pin 36
GPIO.setup(echo_pin, GPIO.IN)  # pin 35
GPIO.output(trigger_pin, GPIO.LOW)


def door_servo_close():
    door_servo.ChangeDutyCycle(12)
    sleep(0.5)
    door_servo.ChangeDutyCycle(0)


def door_servo_open():
    door_servo.ChangeDutyCycle(7)
    sleep(0.5)
    door_servo.ChangeDutyCycle(0)


def distance():
    GPIO.output(trigger_pin, GPIO.HIGH)
    sleep(0.00001)
    GPIO.output(trigger_pin, GPIO.LOW)
    pulse_start_time = 0
    pulse_end_time = 0
    while GPIO.input(echo_pin) == 0:
        pulse_start_time = time()
    while GPIO.input(echo_pin) == 1:
        pulse_end_time = time()
    pulse_duration = pulse_end_time - pulse_start_time
    # speed of sound is 34300 cm/s - however, the signal's path was
    # all the way to the object and back to the sensor, so we need to half it.
    SPEED_OF_SOUND = 34300  # centimeter per second
    distance = pulse_duration * (SPEED_OF_SOUND / 2)
    distance = round(distance, 2)
    print(distance)
    return distance


try:
    door_servo_close()

    start_distance = distance()
    unchanged_distance = 0

    while (1):
        current_distance = distance()

        if abs(current_distance - start_distance) >= 5:
            door_servo_open()
        else:
            sleep(1)
            door_servo_close()

        if abs(current_distance - start_distance) <= 2:
            unchanged_distance += 1
            if unchanged_distance == 10:
                start_distance = current_distance
                unchanged_distance = 0

finally:
    door_servo_close()
    GPIO.cleanup()
