# פרויקט גמר

## שלב 1

**אימון מודל לסיווג מסיכה ב-Google Colab (בעזרת GPU)**

-   מודל מבוסס MobileNet V2:

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x3a_JSMoKCcjiKo2UGFiot2A4JVBdYar?usp=sharing)

## שלב 2

**התקנת מערכת הפעלה Raspberry Pi OS 64-bit על Raspberry Pi 4B**

[יש להוריד את הגירסא האחרונה](https://downloads.raspberrypi.org/raspios_arm64/images/)

מומלץ להתקין על כרטיס זיכרון או דיסק און קי באמצעות [Raspberry Pi Imager](https://www.raspberrypi.org/software/).

לאחר ההתקנה על כרטיס הזיכרון, על מנת לקבל גישה ל-Raspberry Pi בלי לחברו למסך, מקלדת ועכבר חיצוניים יש ליצור קובץ טקסט חדש בשם `ssh` (באותיות קטנות) וללא סיומת בתיקיה הראשית.

על מנת לחבר את ה-Raspberry Pi לרשת האלחוטית המקומית שלנו, ניצור קובץ טקסט חדש שתוכנו:

```txt
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=IL

network={
  ssid="WiFi"
  psk="Password"
}
```

כאשר בתוך המרכאות הכפולות יש למלא את השם והסיסמא לרשת, בהתאמה.

על מנת להתחבר ל-Raspberry Pi על גבי SSH, נתחבר לאותה הרשת שהגדרנו ונבצע את הפקודה הבאה:

```bash
ssh pi@raspberrypi.local
```

הפקודה תבקש סיסמא - הסיסמא ברירת המחדל היא `raspberry` (באותיות קטנות).

## שלב 3

**עדכונים**

לאחר שה-Raspberry Pi נדלק, נבצע מספר פקודות על גבי SSH על מנת לוודא שהמערכת מעודכנת:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt dist-upgrade
sudo apt autoremove --purge
sudo apt clean
```

## שלב 4

**התקנת NumPy**

```bash
pip3 install numpy
```

## שלב 5

**התקנת OpenCV גירסא 4.5.0 (מותאם לארכיטקטורת `arm64`)**

[מדריך של Q-engineering](https://qengineering.eu/install-opencv-4.5-on-raspberry-64-os.html)

## שלב 6

**התקנת TensorFlow Lite Runtime גירסא 2.5.0 (מותאם לארכיטקטורת `arm64`)**

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime
```

[מקור: TendorFlow.org/lite](https://qengineering.eu/install-tensorflow-2.4.0-on-raspberry-64-os.html)

## שלב 7

**התקנת המסך MHS3528**

בשורת הפקודה של ה-Raspberry Pi:

```bash
sudo rm -rf LCD-show
git clone https://github.com/goodtft/LCD-show.git
chmod -R 755 LCD-show
cd LCD-show/
sudo ./MHS35-show
```

## שלב 8

**הורדת הפרויקט**

בשורת הפקודה של הRaspberry Pi:

```bash
git clone https://github.com/lgariv/CollegeProject.git
```

## שלב 9

**הורדת המודל לסיווג המסיכה מ-Google Colab**

העברה ל-Raspberry Pi, פקודה מהמחשב שאליו הורדנו את הקובץ:

```cmd
scp /path/to/model_quant.tflite pi@raspberrypi.local:/home/pi/CollegeProject/models/model_quant.tflite
```

## שלב 10

**הפעלה**

בשורת הפקודה של ה-Raspberry PI:

```bash
cd CollegeProject
python3 door-model.py & python3 Object_detection_webcam_tflite.py && fg
```
