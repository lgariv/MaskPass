# College Project

[עברית](https://github.com/lgariv/CollegeProject/blob/main/README-he.md)

## Step 1

**Train mask image classifier model in Google Colab (with GPU)**

-   Based on MobileNet V2 (SSD) pre-trained model:

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x3a_JSMoKCcjiKo2UGFiot2A4JVBdYar?usp=sharing) _(Download TFLite model after training)_

## Step 2

**Install Raspberry Pi OS 64-bit on RPi 4B**

[Download from here](https://downloads.raspberrypi.org/raspios_arm64/images/)

_Optional -_ before installing, add an empty file called `ssh` (without extension) to the root of the drive, to enable SSH.

## Step 3

**Overclock Raspberry Pi to 1750MHz, overvolt to 3V**

`sudo nano /boot/config.txt`

```txt
arm_freq=1750
over_voltage=3
```

_Optional - enable these lines on `/boot/config.txt` for VNC:_

```txt
hdmi_force_hotplug=1
disable_overscan=1
gpu_mem=128
```

## Step 4

**Post install**

After install, in terminal:

```bash
sudo apt update
sudo apt full-upgrade
```

## Step 5

**Istall OpenCV v4.5.0 or higher (for `arch64`)**

[Q-engineering tutorial](https://qengineering.eu/install-opencv-4.5-on-raspberry-64-os.html)

## Step 6

**Install TensorFlow 2 v2.4.0 or higher (for `arch64`)**

[Q-engineering tutorial](https://qengineering.eu/install-tensorflow-2.4.0-on-raspberry-64-os.html)

Ater install, in terminal:

```bash
sudo apt install protobuf
pip install pycocotools
pip install tf_slim
pip install tensorflow_hub
```

## Step 7

**install protos for TensorFlow 2 Object Detection API**

In terminal:

```bash
chmod +x install-requirements.sh
./install-requirements.sh
```

## Step 8

**Clone repo**

In terminal:

`git clone --recurse-submodules https://github.com/lgariv/CollegeProject.git`

## Step 9

Download TFLite model from Google Colab and place in the same folder

## Step 10

**Run**

In terminal:

```bash
python Object_detection_webcam_tflite.py
```
