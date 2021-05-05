from cv2 import resize, INTER_AREA
import numpy as np
import adafruit_mlx90640
import board, busio

def get_scaled_temp_image():
    def temp_from_random_float(inputNumber, minTemp=35, maxTemp=41):
        return minTemp+inputNumber*(maxTemp-minTemp)
    temp_from_random_float_v = np.vectorize(temp_from_random_float)

    width = 32
    height = 24

    frame_size = (width*height)
    mlx_shape = (height, width)

    random_arr = np.random.rand(frame_size)
    random_arr = temp_from_random_float_v(random_arr)

    frame = None
    try:
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000) # setup I2C
        mlx = adafruit_mlx90640.MLX90640(i2c) # begin MLX90640 with I2C comm
        mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ # set refresh rate

        frame = random_arr
        mlx.getFrame(frame) # read mlx90640
    except:
        print("Could not find or communicate with MLX90640. Using emulated values")
        frame = random_arr

    mlx_image = np.fliplr(np.reshape(frame, mlx_shape))

    resized_mlx_image = resize(mlx_image, (640, 480), interpolation = INTER_AREA)
    return resized_mlx_image


def temp_average_from_bbox(frame, bbox):
    (startX, startY, endX, endY) = bbox
    cropped_image = frame[startY:endY, startX:endX]
    return np.mean(cropped_image)
