sudo apt install -y python3-opencv python3-venv v4l-utils ffmpeg libgpiod2

python3 -m venv venv
source venv/bin/activate
mkdir -p models
pip install --upgrade pip
pip install mediapipe gpiozero RPi.GPIO adafruit-circuitpython-dht
wget -O models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
python HandRecog.py
