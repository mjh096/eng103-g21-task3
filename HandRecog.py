import cv2, os
import mediapipe as mp
from pathlib import Path

from gpiozero import LED

# Variables
WORKING_DIR = Path.cwd()
MODELS_DIR = WORKING_DIR / "models"

# GPIO LEDs
AVAILABLE_LED_PINS = [17,22,27,5,6]
LEDS = {pin: LED(pin) for pin in AVAILABLE_LED_PINS}

# Gesture helpers
TIP_IDS = [4, 8, 12, 16, 20]     # thumb, index, middle, ring, pinky
PIP_IDS = [2, 6, 10, 14, 18]     # thumb IP (2), and PIPs for others

def count_fingers(landmarks, is_right, w, h):
    """Return [thumb,index,middle,ring,pinky] where 1 = up, 0 = down."""
    xs = [int(pt.x * w) for pt in landmarks]
    ys = [int(pt.y * h) for pt in landmarks]
    fingers = [0, 0, 0, 0, 0]

    # Thumb: horizontal check (depends on handedness)
    thumb_tip, thumb_ip = TIP_IDS[0], PIP_IDS[0]
    fingers[0] = 1 if (xs[thumb_tip] > xs[thumb_ip]) == bool(is_right) else 0

    # Other fingers: tip above PIP => up (y grows downwards)
    for i in range(1, 5):
        tip, pip = TIP_IDS[i], PIP_IDS[i]
        fingers[i] = 1 if ys[tip] < ys[pip] else 0
    return fingers

def toggle_gpio_led(pin):
    led = LEDS.get(pin)

    if led is None:
        print(f"WARNING: GPIO {pin} is not configured in AVAILABLE_LED_PINS={AVAILABLE_LED_PINS}")
        return

    try:
        led.toggle()
        print(f"ACTION: GPIO {pin} -> {'ON' if led.is_lit else 'OFF'}")
    except Exception as e:
        print(f"ERROR controlling GPIO {pin}: {e}")
        return bool(led.is_lit) if hasattr(led, "is_lit") else False

def gesture_from_states(f):
    """Map finger states to a label."""
    total = sum(f)
    if total == 0: return "Fist"
    if total == 5: return "Open Palm"
    if total == 1 and f[0] == 1: return "Thumbs Up"
    if total == 1 and f[2] == 1: return "Middle Finger"
    if total == 2 and f[1] == 1 and f[2] == 1: return "Peace"
    if total == 2 and f[1] == 1 and f[4] == 1: return "Rock"
    return "Unknown"

def draw_bottom_label(img, text):
    """Draw a black bar with centered green text at the bottom."""
    h, w = img.shape[:2]
    bar_h = 34
    cv2.rectangle(img, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (w - tw) // 2
    y = h - 10
    cv2.putText(img, text, (x, y), font, scale, (0, 255, 0), thick, cv2.LINE_AA)

#  Camera Settings
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise SystemExit("ERROR: Could not open /dev/video0")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#  MediaPipe Tasks shortcuts
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

MODEL = MODELS_DIR / "hand_landmarker.task"
if not MODEL.exists():
    raise SystemExit(f"Model not found: {MODEL}")

with open(MODEL, "rb") as f:
    model_bytes = f.read()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=model_bytes),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=VisionRunningMode.IMAGE
)

EXCLUDE_GESTURE = {"No Hands", "Unknown"}
CURRENT_GESTURE = "None"

# Main loop
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        label = "No Hands"

        if result.hand_landmarks:
            # Draw landmarks for all hands
            for hand in result.hand_landmarks:
                for pt in hand:
                    x = int(pt.x * w); y = int(pt.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Use first hand to produce a single label (extend if you want per-hand labels)
            first = result.hand_landmarks[0]
            is_right = False
            if result.handedness and len(result.handedness[0]) and hasattr(result.handedness[0][0], "category_name"):
                is_right = (result.handedness[0][0].category_name == "Right")

            states = count_fingers(first, is_right, w, h)
            label = gesture_from_states(states)

            # Action for detected hand gesture
            if label not in EXCLUDE_GESTURE and CURRENT_GESTURE != label:
                CURRENT_GESTURE = label
                print("Detected Gesture: ", label)

                match label:
                    case "Rock":
                        toggle_gpio_led(27)
                    case "Peace":
                        toggle_gpio_led(17)
                    case "Middle Finger":
                        toggle_gpio_led(22)
                    case "Thumbs Up":
                        toggle_gpio_led(5)
                    case "Open Palm":
                        toggle_gpio_led(6)

        # Add Bottom label
        draw_bottom_label(frame, label)

        cv2.imshow("Hands", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):  # q or ESC
            break
        if cv2.getWindowProperty("Hands", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()