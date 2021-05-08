#
# https://pimylifeup.com/raspberry-pi-distance-sensor/
#
import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)
GPIO.setup(32, GPIO.OUT)
door_servo = GPIO.PWM(32, 50)  # pin 32 for servo, operating at 50Hz
door_servo.start(0)  # start servo pulse with 0% duty cycle (disabled)

trigger_pin = 36
echo_pin = 35

GPIO.setup(trigger_pin, GPIO.OUT)  # pin 36
GPIO.setup(echo_pin, GPIO.IN)  # pin 35
GPIO.output(trigger_pin, GPIO.LOW)

door_is_opened = False


def door_servo_close():
    if door_is_opened:
        door_servo.ChangeDutyCycle(7)
        sleep(0.5)
        door_servo.ChangeDutyCycle(0)


def door_servo_open():
    if not door_is_opened:
        door_servo.ChangeDutyCycle(2)
        sleep(0.5)
        door_servo.ChangeDutyCycle(0)


def distance():
    GPIO.output(trigger_pin, GPIO.HIGH)
    sleep(0.00001)
    GPIO.output(trigger_pin, GPIO.LOW)
    while GPIO.input(echo_pin) == 0:
        pulse_start_time = time()
    while GPIO.input(echo_pin) == 1:
        pulse_end_time = time()
    pulse_duration = pulse_end_time - pulse_start_time
    # speed of sound is 34300 cm/s - however, the signal's path was
    # all the way to the object and back to the sensor, so we need to half it.
    distance = round(pulse_duration * 17150, 2)
    return distance


try:
    door_servo_close()

    last_distance = distance()
    unchanged_distance = 0

    threshold_distance = distance() - 0.1

    while (1):
        current_distance = distance()

        if current_distance <= threshold_distance:
            door_servo_open()
        else:
            sleep(1)
            door_servo_close()

        if abs(current_distance - last_distance) <= 1:
            unchanged_distance += 1
            if unchanged_distance == 10:
                threshold_distance = current_distance - 0.1

        last_distance = current_distance
finally:
    door_servo_close()
    GPIO.cleanup()
