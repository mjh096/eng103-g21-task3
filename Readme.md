sudo apt install -y python3-opencv python3-venv v4l-utils ffmpeg
python3 -m venv ~/vpihand && source ~/vpihand/bin/activate
mkdir -p ~/models
wget -O ~/models/hand_landmarker.task   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
python HandRecog.py