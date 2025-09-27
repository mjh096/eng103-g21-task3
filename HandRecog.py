"""Hand gesture recognition loop using MediaPipe Tasks to drive GPIO LEDs.

The script is designed for Raspberry Pi deployments where gestures from a camera
feed are mapped to simple GPIO outputs (e.g. LEDs). It falls back between USB
and Pi cameras and renders feedback via an OpenCV window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence
import time
import board
import adafruit_dht
import cv2
import mediapipe as mp
from gpiozero import LED
from mediapipe.framework.formats import landmark_pb2

# Directories and model assets
WORKING_DIR = Path.cwd()
MODELS_DIR = WORKING_DIR / "models"
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# GPIO configuration
AVAILABLE_LED_PINS = (17, 22, 27, 5, 6)
LEDS: Dict[int, LED] = {pin: LED(pin) for pin in AVAILABLE_LED_PINS}
GESTURE_TO_PIN = {
    "Rock": 27,
    "Peace": 17,
    "Middle Finger": 22,
    "Thumbs Up": 5,
    "Open Palm": 6,
}

# DHT Sensor configuration
DHT_BCM_PIN = 4

# Gesture helpers
TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
PIP_IDS = [2, 6, 10, 14, 18]  # thumb IP (2), and PIPs for others

EXCLUDE_GESTURE = {"No Hands", "Unknown"}

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

def _board_pin_from_bcm(bcm: int):
    """Map a BCM pin number (e.g., 4) to the corresponding `board.D#` pin (e.g., board.D4)."""
    try:
        return getattr(board, f"D{bcm}")
    except AttributeError as e:
        raise ValueError(f"No board pin mapping for BCM {bcm}.") from e

# Create one reusable sensor instance
_dht = adafruit_dht.DHT11(_board_pin_from_bcm(DHT_BCM_PIN), use_pulseio=False)

def set_dht_pin(bcm_pin: int) -> None:
    """Reinitialize the DHT11 on a new BCM pin at runtime."""
    global _dht, DHT_BCM_PIN
    DHT_BCM_PIN = bcm_pin
    _dht = adafruit_dht.DHT11(_board_pin_from_bcm(bcm_pin), use_pulseio=False)

def read_dht11(pin: int = DHT_BCM_PIN) -> tuple[float | None, float | None]:
    """Read humidity (%) and temperature (°C) from a DHT11 sensor."""
    global _dht
    # Re-init if caller specifies a different pin
    if pin != DHT_BCM_PIN:
        set_dht_pin(pin)

    try:
        humidity = _dht.humidity
        temperature = _dht.temperature
        if humidity is None or temperature is None:
            print("WARNING: DHT11 read returned None values.")
        return humidity, temperature
    except RuntimeError as e:
        # Common/expected for DHT sensors; just try again later
        print(f"WARNING: DHT11 read failed: {e}")
        return None, None
    finally:
        time.sleep(2.0)

def read_dht11_temperature(pin: int = DHT_BCM_PIN) -> float | None:
    """Read only the temperature (°C) from the DHT11."""
    _, temperature = read_dht11(pin)
    return temperature


def count_fingers(
    landmarks: Sequence[landmark_pb2.NormalizedLandmark],
    is_right: bool,
    width: int,
    height: int,
) -> list[int]:
    """Return a five-element list indicating which fingers are extended."""
    xs = [int(point.x * width) for point in landmarks]
    ys = [int(point.y * height) for point in landmarks]
    fingers = [0, 0, 0, 0, 0]

    thumb_tip, thumb_ip = TIP_IDS[0], PIP_IDS[0]
    fingers[0] = 1 if (xs[thumb_tip] > xs[thumb_ip]) == bool(is_right) else 0

    for i in range(1, 5):
        tip, pip = TIP_IDS[i], PIP_IDS[i]
        fingers[i] = 1 if ys[tip] < ys[pip] else 0
    return fingers


def toggle_gpio_led(pin: int) -> bool:
    """Toggle the LED on *pin* and return the new lit state."""
    led = LEDS.get(pin)
    if led is None:
        print(
            f"WARNING: GPIO {pin} is not configured in AVAILABLE_LED_PINS={AVAILABLE_LED_PINS}"
        )
        return False

    try:
        led.toggle()
        print(f"ACTION: GPIO {pin} -> {'ON' if led.is_lit else 'OFF'}")
    except Exception as exc:  # pragma: no cover - hardware specific
        print(f"ERROR controlling GPIO {pin}: {exc}")
    return bool(getattr(led, "is_lit", False))


def gesture_from_states(finger_states: Sequence[int]) -> str:
    """Map finger states to a descriptive label."""
    total = sum(finger_states)
    if total == 0:
        return "Fist"
    if total == 5:
        return "Open Palm"
    if total == 1 and finger_states[0] == 1:
        return "Thumbs Up"
    if total == 1 and finger_states[2] == 1:
        return "Middle Finger"
    if total == 2 and finger_states[1] == 1 and finger_states[2] == 1:
        return "Peace"
    if total == 2 and finger_states[1] == 1 and finger_states[4] == 1:
        return "Rock"
    return "Unknown"


def draw_bottom_label(image, text: str) -> None:
    """Draw a black bar with centered green text at the bottom of the frame."""
    height, width = image.shape[:2]
    bar_height = 34
    cv2.rectangle(image, (0, height - bar_height), (width, height), (0, 0, 0), -1)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (text_width, _), _ = cv2.getTextSize(text, font, scale, thick)
    x = (width - text_width) // 2
    y = height - 10
    cv2.putText(image, text, (x, y), font, scale, (0, 255, 0), thick, cv2.LINE_AA)


class UnifiedCamera:
    """Handle camera access across USB webcams and PiCamera2."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.mode: str | None = None
        self.cap = None
        self.picam2 = None

        self._initialise_camera()

    def _initialise_camera(self) -> None:
        """Initialise a USB camera first, falling back to PiCamera2."""
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                self.mode = "cv2"
                self.cap = cap
                print("Camera: Using USB webcam (/dev/video0) via OpenCV.")
                return
            cap.release()

        try:  # pragma: no cover - hardware specific
            from picamera2 import Picamera2

            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"format": "RGB888", "size": (self.width, self.height)},
                controls={
                    "FrameDurationLimits": (
                        int(1e6 / self.fps),
                        int(1e6 / self.fps),
                    )
                },
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.mode = "picam2"
            print("Camera: Using Pi Camera v2 via Picamera2.")
        except Exception as exc:
            raise SystemExit(
                "ERROR: No camera available (USB failed, Picamera2 fallback failed: "
                f"{exc})"
            )

    def read(self) -> tuple[bool, object | None]:
        """Return a frame tuple (success flag, frame)."""
        if self.mode == "cv2" and self.cap is not None:
            return self.cap.read()
        if self.mode == "picam2" and self.picam2 is not None:  # pragma: no cover

            frame_rgb = self.picam2.capture_array()
            if frame_rgb is None:
                return False, None
            return True, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return False, None

    def release(self) -> None:
        """Release camera resources regardless of the backend used."""
        if self.mode == "cv2" and self.cap is not None:
            self.cap.release()
        elif self.mode == "picam2" and self.picam2 is not None:  # pragma: no cover
            try:
                self.picam2.stop()
            except Exception:
                pass


def load_model_bytes(model_path: Path) -> bytes:
    """Read the MediaPipe hand landmark model from disk."""
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    return model_path.read_bytes()


def create_landmarker_options(model_bytes: bytes) -> HandLandmarkerOptions:
    """Build task options for running hand landmark inference on frames."""
    return HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=VisionRunningMode.IMAGE,
    )


def handle_gesture(label: str) -> None:
    """Toggle GPIO output corresponding to *label* if configured."""
    pin = GESTURE_TO_PIN.get(label)
    if pin is None:
        return
    toggle_gpio_led(pin)


def main() -> None:
    """Entry-point for gesture detection and GPIO feedback loop."""
    camera = UnifiedCamera(width=640, height=480, fps=30)

    model_bytes = load_model_bytes(MODEL_PATH)
    options = create_landmarker_options(model_bytes)

    current_label = "None"

    with HandLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                success, frame = camera.read()
                if not success or frame is None:
                    print("Camera read failed; exiting loop.")
                    break

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                label = "No Hands"

                if result.hand_landmarks:
                    for hand_landmarks in result.hand_landmarks:
                        for landmark in hand_landmarks:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    first_hand = result.hand_landmarks[0]
                    is_right = False
                    if (
                        result.handedness
                        and len(result.handedness[0])
                        and hasattr(result.handedness[0][0], "category_name")
                    ):
                        is_right = result.handedness[0][0].category_name == "Right"

                    finger_states = count_fingers(first_hand, is_right, width, height)
                    label = gesture_from_states(finger_states)

                    if label not in EXCLUDE_GESTURE and label != current_label:
                        current_label = label
                        print(f"Detected Gesture: {label}")
                        handle_gesture(label)

                draw_bottom_label(frame, label)

                cv2.imshow("Hands", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty("Hands", cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



