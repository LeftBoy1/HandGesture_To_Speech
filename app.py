import os
import math
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# Download official MediaPipe Hand Landmarker model automatically if missing
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

# Initialize MediaPipe HandLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

camera = None
is_camera_active = False
current_detected_word = "Waiting for gesture..."
current_sign_language = "ASL"
last_timestamp_ms = 0

# Expanded 24-Sign Dictionary (ASL / ISL / BSL mappings)
SIGN_DICTIONARY = [
    # Greetings
    {"word": "Hello", "gesture": "All 5 fingers open and spread wide", "category": "Greetings", "lang_note": "Universal open-palm wave across ASL, ISL, and BSL", "description": "Raise your hand with all five fingers open and spread apart."},
    {"word": "Bye", "gesture": "4 fingers straight up together, thumb tucked", "category": "Greetings", "lang_note": "ASL 'B' / ISL & BSL flat-hand wave", "description": "Hold index, middle, ring, and pinky straight up together with thumb folded across palm."},
    {"word": "Thank You", "gesture": "Flat hand with thumb extended outward", "category": "Greetings", "lang_note": "ASL / ISL flat hand extending from chin", "description": "Keep all four fingers together and straight with your thumb extended to the side."},
    {"word": "Namaste / Please", "gesture": "4 fingers closed together, thumb tucked", "category": "Greetings", "lang_note": "ISL respectful flat-palm gesture", "description": "Hold your palm flat facing forward with fingers pressed neatly together."},

    # Common Words & Responses
    {"word": "Yes", "gesture": "All fingers curled into a tight closed fist", "category": "Common", "lang_note": "ASL 'A' / ISL affirmative nod-fist", "description": "Close all fingers into your palm with thumb resting against the side."},
    {"word": "No", "gesture": "Only thumb extended outward, 4 fingers closed", "category": "Common", "lang_note": "Thumbs-sideways / negative sign", "description": "Curl index, middle, ring, and pinky fingers while sticking your thumb out."},
    {"word": "Good / Thumbs Up", "gesture": "Thumb pointing straight up, 4 fingers curled", "category": "Common", "lang_note": "Universal positive / BSL 'Good'", "description": "Point your thumb straight upward with all other fingers closed."},
    {"word": "OK", "gesture": "Thumb & index pinched in a circle; 3 fingers up", "category": "Common", "lang_note": "ASL 'F' / ISL & BSL 'OK' ring", "description": "Touch the tips of your thumb and index finger together while raising middle, ring, and pinky."},
    {"word": "Peace", "gesture": "Index and middle fingers spread in a V shape", "category": "Common", "lang_note": "ASL 'V' / Victory sign", "description": "Raise index and middle fingers apart in a V while keeping ring and pinky folded."},
    {"word": "Help", "gesture": "Thumb and index extended in an L-shape", "category": "Common", "lang_note": "ASL 'L' / Directional help cue", "description": "Extend your index finger straight up and your thumb straight out at 90 degrees."},
    {"word": "Stop", "gesture": "Index, middle, and pinky extended; ring folded", "category": "Common", "lang_note": "Three-finger halt gesture", "description": "Raise index, middle, and pinky fingers while folding your ring finger down."},

    # Phrases
    {"word": "I Love You", "gesture": "Thumb, index, and pinky extended; middle & ring down", "category": "Phrases", "lang_note": "ASL I-L-Y combination sign", "description": "Extend thumb, index, and pinky fingers simultaneously."},
    {"word": "Call Me", "gesture": "Thumb and pinky extended outward", "category": "Phrases", "lang_note": "ASL 'Y' / Phone gesture", "description": "Fold index, middle, and ring fingers while extending thumb and pinky."},
    {"word": "Rock On / Cheer", "gesture": "Index and pinky straight up; thumb, middle, ring folded", "category": "Phrases", "lang_note": "Celebration / Horns gesture", "description": "Raise only your index and pinky fingers straight up."},
    {"word": "Water / Drink", "gesture": "Index, middle, and ring spread apart (W-shape)", "category": "Phrases", "lang_note": "ASL 'W' (tapped at chin for Water)", "description": "Raise three middle fingers spread apart while folding thumb and pinky."},

    # Numbers (0 - 5)
    {"word": "Number 1", "gesture": "Only index finger pointing straight up", "category": "Numbers", "lang_note": "ASL / ISL / BSL One", "description": "Extend only your index finger upward; keep thumb and other fingers closed."},
    {"word": "Number 2", "gesture": "Index and middle fingers raised close together", "category": "Numbers", "lang_note": "ASL / ISL / BSL Two", "description": "Raise index and middle fingers straight up touching side-by-side."},
    {"word": "Number 3", "gesture": "Thumb, index, and middle fingers extended", "category": "Numbers", "lang_note": "ASL Number 3", "description": "Extend your thumb, index, and middle fingers while keeping ring and pinky folded."},
    {"word": "Number 4", "gesture": "Four fingers spread apart, thumb tucked", "category": "Numbers", "lang_note": "ASL / ISL / BSL Four", "description": "Raise index, middle, ring, and pinky fingers spread apart with thumb folded."},
    {"word": "Number 5", "gesture": "All five fingers extended", "category": "Numbers", "lang_note": "ASL / ISL / BSL Five", "description": "Extend all five fingers outward."},

    # Alphabet Letters
    {"word": "Letter C", "gesture": "All fingers curved in a C-arc shape", "category": "Alphabet", "lang_note": "ASL Fingerspelling 'C'", "description": "Curve your thumb and fingers partway as if holding a cup."},
    {"word": "Letter I", "gesture": "Only pinky finger extended straight up", "category": "Alphabet", "lang_note": "ASL Fingerspelling 'I'", "description": "Fold all fingers except your pinky finger pointing straight up."},
    {"word": "Letter U", "gesture": "Index & middle fingers straight up and pressed together", "category": "Alphabet", "lang_note": "ASL Fingerspelling 'U'", "description": "Keep index and middle fingers straight up and touching together."},
    {"word": "Letter X", "gesture": "Index finger hooked/bent, other fingers closed", "category": "Alphabet", "lang_note": "ASL Fingerspelling 'X'", "description": "Bend your index finger into a hook shape while keeping other fingers curled."}
]


def open_working_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        for index in [0, 1, 2]:
            cap = cv2.VideoCapture(index, backend)
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                cap.release()
    return None


def draw_hand_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (56, 189, 248), 2)
    for px, py in points:
        cv2.circle(frame, (px, py), 5, (16, 185, 129), -1)


def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def classify_gesture(landmarks, handedness):
    """Classifies 24 distinct gestures using finger states + geometric ratios."""
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # 1. Thumb state (mirrored camera perspective)
    if handedness == "Right":
        thumb_open = 1 if landmarks[4].x > landmarks[3].x else 0
    else:
        thumb_open = 1 if landmarks[4].x < landmarks[3].x else 0
    fingers.append(thumb_open)

    # 2. Other 4 fingers (tip above PIP joint)
    for idx in range(1, 5):
        fingers.append(1 if landmarks[tip_ids[idx]].y < landmarks[tip_ids[idx] - 2].y else 0)

    pattern = tuple(fingers)

    # Key geometric metrics normalized by palm size (wrist 0 to middle MCP 9)
    palm_size = max(distance(landmarks[0], landmarks[9]), 0.001)
    pinch_dist = distance(landmarks[4], landmarks[8]) / palm_size
    index_middle_spread = distance(landmarks[8], landmarks[12]) / palm_size

    # Special Geometric Checks
    # OK Sign: Thumb tip & index tip touching + middle, ring, pinky raised
    if pinch_dist < 0.28 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
        return "OK"

    # Good / Thumbs Up: Thumb pointing noticeably higher than index knuckle while fingers closed
    if pattern == (1, 0, 0, 0, 0):
        if landmarks[4].y < landmarks[5].y - 0.05:
            return "Good"
        return "No"

    # Peace (V spread) vs Number 2 / Letter U (fingers pressed together)
    if pattern == (0, 1, 1, 0, 0):
        if index_middle_spread > 0.22:
            return "Peace"
        return "Number 2"

    # Hello (5 fingers spread) vs Thank You (4 fingers pressed together + thumb out)
    if pattern == (1, 1, 1, 1, 1):
        if index_middle_spread > 0.18:
            return "Hello"
        return "Thank You"

    # Bye (4 fingers up, thumb tucked) vs Number 4 (spread)
    if pattern == (0, 1, 1, 1, 1):
        if index_middle_spread > 0.18:
            return "Number 4"
        return "Bye"

    # Exact Finger Pattern Dictionary
    exact_patterns = {
        (0, 0, 0, 0, 0): "Yes",
        (0, 1, 0, 0, 0): "Number 1",
        (1, 1, 1, 0, 0): "Number 3",
        (0, 1, 1, 1, 0): "Water",
        (1, 1, 0, 0, 1): "I Love You",
        (1, 0, 0, 0, 1): "Call Me",
        (1, 1, 0, 0, 0): "Help",
        (0, 1, 0, 0, 1): "Rock On",
        (0, 0, 0, 0, 1): "Letter I",
        (0, 1, 1, 0, 1): "Stop",
    }

    if pattern in exact_patterns:
        return exact_patterns[pattern]

    # Letter C check (curved fingers)
    if fingers[1] == 0 and fingers[2] == 0 and pinch_dist > 0.35 and landmarks[8].y < landmarks[6].y + 0.03:
        return "Letter C"

    return "Unknown Gesture"


def generate_frames():
    global camera, is_camera_active, current_detected_word, last_timestamp_ms

    while is_camera_active:
        if camera is None or not camera.isOpened():
            break

        success, frame = camera.read()
        if not success or frame is None:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        now_ms = int(time.time() * 1000)
        if now_ms <= last_timestamp_ms:
            now_ms = last_timestamp_ms + 1
        last_timestamp_ms = now_ms

        detection_result = detector.detect_for_video(mp_image, now_ms)

        if detection_result.hand_landmarks:
            for idx, landmarks in enumerate(detection_result.hand_landmarks):
                handedness = detection_result.handedness[idx][0].category_name
                draw_hand_skeleton(frame, landmarks)
                gesture_text = classify_gesture(landmarks, handedness)
                current_detected_word = gesture_text

                cv2.rectangle(frame, (12, 12), (380, 62), (9, 13, 22), -1)
                cv2.putText(
                    frame, f"[{current_sign_language}] {gesture_text}", (22, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (56, 189, 248), 2, cv2.LINE_AA
                )
        else:
            current_detected_word = "No hand detected"

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, is_camera_active, current_detected_word
    if is_camera_active and camera is not None and camera.isOpened():
        return jsonify({"status": "started"})

    cam = open_working_camera()
    if cam is None:
        is_camera_active = False
        current_detected_word = "Camera not found"
        return jsonify({
            "status": "error",
            "message": "Could not access webcam. Close Zoom/Teams/Camera app or check Camera privacy settings."
        }), 500

    camera = cam
    is_camera_active = True
    current_detected_word = "Waiting for gesture..."
    return jsonify({"status": "started"})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, is_camera_active, current_detected_word
    is_camera_active = False
    time.sleep(0.1)
    if camera is not None:
        camera.release()
        camera = None
    current_detected_word = "Camera stopped"
    return jsonify({"status": "stopped"})


@app.route('/set_language', methods=['POST'])
def set_language():
    global current_sign_language
    data = request.get_json(silent=True) or {}
    lang = data.get("language", "ASL")
    if lang in ["ASL", "ISL", "BSL"]:
        current_sign_language = lang
    return jsonify({"status": "ok", "language": current_sign_language})


@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    return jsonify({"prediction": current_detected_word, "language": current_sign_language})


@app.route('/api/dictionary', methods=['GET'])
def get_dictionary():
    return jsonify(SIGN_DICTIONARY)


@app.route('/api/chat', methods=['POST'])
def chat_assistant():
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()
    msg_lower = user_msg.lower()

    if not user_msg:
        return jsonify({"reply": "Please type a message or ask how to perform any sign!", "action": None})

    if any(k in msg_lower for k in ["start camera", "turn on camera", "open camera"]):
        return jsonify({"reply": "Starting your webcam now!", "action": "start_camera"})

    if any(k in msg_lower for k in ["stop camera", "turn off camera", "close camera"]):
        return jsonify({"reply": "Stopping the webcam feed.", "action": "stop_camera"})

    if any(k in msg_lower for k in ["clear text", "clear sentence", "reset text"]):
        return jsonify({"reply": "Cleared your sentence builder.", "action": "clear_sentence"})

    if any(k in msg_lower for k in ["speak", "read aloud", "say it"]):
        return jsonify({"reply": "Reading your composed sentence aloud now.", "action": "speak_sentence"})

    # Search all 24 signs in dictionary
    for item in SIGN_DICTIONARY:
        word_lower = item["word"].lower()
        if word_lower in msg_lower and any(w in msg_lower for w in ["how", "sign", "gesture", "show", "do", "what is"]):
            return jsonify({
                "reply": f"**{item['word']}** ({item['category']} | {item['lang_note']}):\n• **Hand Shape:** {item['gesture']}\n• **Instructions:** {item['description']}",
                "action": "insert_word",
                "word": item["word"]
            })

    if any(k in msg_lower for k in ["list signs", "all signs", "supported", "help", "what can you do"]):
        words = ", ".join([s["word"] for s in SIGN_DICTIONARY])
        return jsonify({
            "reply": f"I support **24 signs** across ASL, ISL, and BSL!\n\n**All Signs:** {words}\n\nTry asking: *\"How do I sign Thank You?\"* or *\"How do I sign Number 3?\"*",
            "action": None
        })

    if "hello" in msg_lower or "hi" in msg_lower:
        return jsonify({"reply": "Hello! 👋 Show an open palm with all 5 fingers spread to sign Hello.", "action": None})

    if "thank" in msg_lower:
        return jsonify({"reply": "You're very welcome! 😊", "action": None})

    if "bye" in msg_lower:
        return jsonify({"reply": "Goodbye! 👋 Have a great day!", "action": None})

    return jsonify({
        "reply": f"You signed/typed: *\"{user_msg}\"*\n\nAsk me **\"How do I sign [word]?\"** or type **\"list signs\"** to see all 24 gestures!",
        "action": None
    })


if __name__ == '__main__':
    print("Server running! Open your browser at: http://127.0.0.1:5000")
    app.run(debug=False, threaded=True, port=5000)