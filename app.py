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
recognition_mode = "words"  # "words" or "alphabet"
last_timestamp_ms = 0

# Words & Phrases Dictionary
WORD_DICTIONARY = [
    {"word": "Hello", "gesture": "All 5 fingers open and spread wide", "category": "Greetings", "lang_note": "ASL / ISL / BSL", "description": "Raise your hand with all five fingers open and spread apart."},
    {"word": "Bye", "gesture": "4 fingers straight up together, thumb tucked", "category": "Greetings", "lang_note": "ASL / ISL / BSL", "description": "Hold index, middle, ring, and pinky straight up together with thumb folded across palm."},
    {"word": "Thank You", "gesture": "Flat hand with thumb extended outward", "category": "Greetings", "lang_note": "ASL / ISL", "description": "Keep all four fingers together and straight with your thumb extended to the side."},
    {"word": "Yes", "gesture": "All fingers curled into a tight closed fist", "category": "Common", "lang_note": "ASL / ISL", "description": "Close all fingers into your palm with thumb resting against the side."},
    {"word": "No", "gesture": "Only thumb extended outward, 4 fingers closed", "category": "Common", "lang_note": "Universal", "description": "Curl index, middle, ring, and pinky fingers while sticking your thumb out."},
    {"word": "Good", "gesture": "Thumb pointing straight up, 4 fingers curled", "category": "Common", "lang_note": "Universal / BSL", "description": "Point your thumb straight upward with all other fingers closed."},
    {"word": "OK", "gesture": "Thumb & index pinched in a circle; 3 fingers up", "category": "Common", "lang_note": "ASL / ISL / BSL", "description": "Touch thumb and index fingertips together while raising middle, ring, and pinky."},
    {"word": "Peace", "gesture": "Index and middle fingers spread in a V shape", "category": "Common", "lang_note": "ASL / ISL", "description": "Raise index and middle fingers apart in a V while keeping ring and pinky folded."},
    {"word": "Help", "gesture": "Thumb and index extended in an L-shape", "category": "Common", "lang_note": "ASL", "description": "Extend index finger straight up and thumb straight out at 90 degrees."},
    {"word": "I Love You", "gesture": "Thumb, index, and pinky extended; middle & ring down", "category": "Phrases", "lang_note": "ASL I-L-Y", "description": "Extend thumb, index, and pinky fingers simultaneously."},
    {"word": "Call Me", "gesture": "Thumb and pinky extended outward", "category": "Phrases", "lang_note": "ASL", "description": "Fold index, middle, and ring fingers while extending thumb and pinky."},
    {"word": "Water", "gesture": "Index, middle, and ring spread apart (W-shape)", "category": "Phrases", "lang_note": "ASL", "description": "Raise three middle fingers spread apart while folding thumb and pinky."},
]

# Full A-Z English Alphabet Fingerspelling Dictionary
ALPHABET_DICTIONARY = [
    {"word": "A", "gesture": "Closed fist with thumb resting upright on the side of index finger", "category": "Alphabet", "lang_note": "ASL Letter A", "description": "Make a fist with all fingers curled and thumb straight alongside the index finger."},
    {"word": "B", "gesture": "4 fingers straight up pressed together, thumb folded across palm", "category": "Alphabet", "lang_note": "ASL Letter B", "description": "Hold index, middle, ring, and pinky straight up together; tuck thumb across palm."},
    {"word": "C", "gesture": "Thumb and fingers curved in a C-arc shape", "category": "Alphabet", "lang_note": "ASL Letter C", "description": "Curve your hand into a C shape as if holding a cup."},
    {"word": "D", "gesture": "Index finger straight up; thumb touches middle, ring & pinky tips", "category": "Alphabet", "lang_note": "ASL Letter D", "description": "Point index finger straight up while forming a loop with thumb and remaining fingers."},
    {"word": "E", "gesture": "All fingertips curled down onto the folded thumb", "category": "Alphabet", "lang_note": "ASL Letter E", "description": "Bend all four fingers down so their tips rest above your tucked thumb."},
    {"word": "F", "gesture": "Thumb & index pinched in a circle; middle, ring, pinky straight up", "category": "Alphabet", "lang_note": "ASL Letter F", "description": "Touch index tip to thumb tip and spread the other three fingers upward."},
    {"word": "G", "gesture": "Index finger and thumb pointing horizontally to the side", "category": "Alphabet", "lang_note": "ASL Letter G", "description": "Point your index finger and thumb parallel horizontally."},
    {"word": "H", "gesture": "Index and middle fingers pointing horizontally together", "category": "Alphabet", "lang_note": "ASL Letter H", "description": "Extend index and middle fingers side-by-side pointing horizontally."},
    {"word": "I", "gesture": "Only pinky finger extended straight up, fist closed", "category": "Alphabet", "lang_note": "ASL Letter I", "description": "Make a fist and raise only your pinky finger straight up."},
    {"word": "J", "gesture": "Pinky extended while tilting hand inward (J-curve)", "category": "Alphabet", "lang_note": "ASL Letter J", "description": "Extend pinky finger and scoop/tilt wrist inward to trace a J."},
    {"word": "K", "gesture": "Index & middle up in a V with thumb tucked between them", "category": "Alphabet", "lang_note": "ASL Letter K", "description": "Raise index and middle fingers slightly apart and place thumb tip between them."},
    {"word": "L", "gesture": "Index straight up and thumb straight out forming an L", "category": "Alphabet", "lang_note": "ASL Letter L", "description": "Extend thumb and index finger at a 90-degree L angle."},
    {"word": "M", "gesture": "3 fingers (index, middle, ring) folded over tucked thumb", "category": "Alphabet", "lang_note": "ASL Letter M", "description": "Tuck thumb under index, middle, and ring fingers."},
    {"word": "N", "gesture": "2 fingers (index, middle) folded over tucked thumb", "category": "Alphabet", "lang_note": "ASL Letter N", "description": "Tuck thumb under index and middle fingers."},
    {"word": "O", "gesture": "All fingertips curved to touch thumb tip in an O ring", "category": "Alphabet", "lang_note": "ASL Letter O", "description": "Bring all fingertips together to meet the thumb tip in a circle."},
    {"word": "P", "gesture": "Index & middle pointing downward with thumb between", "category": "Alphabet", "lang_note": "ASL Letter P", "description": "Point your K-handshape downward toward the floor."},
    {"word": "Q", "gesture": "Index finger and thumb pointing straight downward", "category": "Alphabet", "lang_note": "ASL Letter Q", "description": "Point your G-handshape downward toward the floor."},
    {"word": "R", "gesture": "Index and middle fingers crossed over each other", "category": "Alphabet", "lang_note": "ASL Letter R", "description": "Raise index and middle fingers and cross middle finger behind index."},
    {"word": "S", "gesture": "Tight closed fist with thumb wrapped across the front of fingers", "category": "Alphabet", "lang_note": "ASL Letter S", "description": "Make a fist and wrap your thumb directly across the front of your curled fingers."},
    {"word": "T", "gesture": "Fist closed with thumb poking up between index and middle fingers", "category": "Alphabet", "lang_note": "ASL Letter T", "description": "Tuck your thumb tip out between your folded index and middle fingers."},
    {"word": "U", "gesture": "Index and middle fingers straight up pressed tightly together", "category": "Alphabet", "lang_note": "ASL Letter U", "description": "Hold index and middle fingers upright and touching side-by-side."},
    {"word": "V", "gesture": "Index and middle fingers straight up spread apart in a V", "category": "Alphabet", "lang_note": "ASL Letter V", "description": "Hold index and middle fingers upright and separated in a V."},
    {"word": "W", "gesture": "Index, middle, and ring fingers spread upward", "category": "Alphabet", "lang_note": "ASL Letter W", "description": "Raise index, middle, and ring fingers spread apart; fold thumb and pinky."},
    {"word": "X", "gesture": "Index finger bent like a hook, other fingers closed", "category": "Alphabet", "lang_note": "ASL Letter X", "description": "Curl your index finger into a hook while keeping other fingers closed."},
    {"word": "Y", "gesture": "Thumb and pinky extended outward, 3 middle fingers closed", "category": "Alphabet", "lang_note": "ASL Letter Y", "description": "Extend thumb and pinky finger outward while folding middle three fingers."},
    {"word": "Z", "gesture": "Index finger pointing diagonally across (Z-angle)", "category": "Alphabet", "lang_note": "ASL Letter Z", "description": "Extend index finger diagonally across your palm to trace a Z."}
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


def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def classify_alphabet(lm, handedness):
    """Reliable A-Z ASL Alphabet classifier invariant to camera mirroring."""
    palm = max(dist(lm[0], lm[9]), 0.001)

    # Finger extension using distance from wrist (more reliable than raw y-axis alone)
    idx_ext = dist(lm[8], lm[0]) / palm > 1.35 and lm[8].y < lm[6].y
    mid_ext = dist(lm[12], lm[0]) / palm > 1.35 and lm[12].y < lm[10].y
    rng_ext = dist(lm[16], lm[0]) / palm > 1.30 and lm[16].y < lm[14].y
    pnk_ext = dist(lm[20], lm[0]) / palm > 1.20 and lm[20].y < lm[18].y

    # Downward pointing (P and Q)
    idx_down = lm[8].y > lm[5].y + (0.45 * palm)
    mid_down = lm[12].y > lm[9].y + (0.45 * palm)

    # Horizontal pointing (G and H)
    idx_horiz = abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y) * 1.25 and dist(lm[8], lm[5]) / palm > 0.50
    mid_horiz = abs(lm[12].x - lm[9].x) > abs(lm[12].y - lm[9].y) * 1.25 and dist(lm[12], lm[9]) / palm > 0.50

    # Mirror-proof thumb check: compare thumb tip distance from pinky knuckle (17) vs index knuckle (5)
    thumb_to_pinky_mcp = dist(lm[4], lm[17]) / palm
    thumb_to_index_mcp = dist(lm[4], lm[5]) / palm
    thumb_out = thumb_to_pinky_mcp > 0.85 and thumb_to_index_mcp > 0.42

    pinch_index = dist(lm[4], lm[8]) / palm
    pinch_mid = dist(lm[4], lm[12]) / palm
    idx_mid_spread = dist(lm[8], lm[12]) / palm

    # 1. Downward Letters: P, Q
    if idx_down and mid_down and not rng_ext and not pnk_ext:
        return "P"
    if idx_down and not mid_down and not rng_ext and not pnk_ext:
        return "Q"

    # 2. Horizontal Letters: G, H
    if idx_horiz and mid_horiz and not rng_ext and not pnk_ext:
        return "H"
    if idx_horiz and not mid_horiz and not rng_ext and not pnk_ext:
        return "G"

    # 3. All 4 Fingers Straight Up: B
    if idx_ext and mid_ext and rng_ext and pnk_ext:
        return "B"

    # 4. Three Fingers Up: F (thumb-index pinch) vs W (spread)
    if not idx_ext and mid_ext and rng_ext and pnk_ext:
        return "F"
    if idx_ext and mid_ext and rng_ext and not pnk_ext:
        return "W"

    # 5. Two Fingers Up (Index + Middle): R, K, V, U
    if idx_ext and mid_ext and not rng_ext and not pnk_ext:
        # Crossed fingers (R): index tip is closer to ring knuckle (13) than middle tip is
        if dist(lm[8], lm[13]) < dist(lm[12], lm[13]) - 0.015:
            return "R"
        # Thumb raised between index & middle (K)
        if lm[4].y < lm[6].y and idx_mid_spread > 0.20:
            return "K"
        return "V" if idx_mid_spread > 0.24 else "U"

    # 6. Index Finger Only: L, Z, D
    if idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        if thumb_out:
            return "L"
        if abs(lm[8].x - lm[5].x) / palm > 0.38:
            return "Z"
        return "D"

    # 7. Pinky Finger Only: Y, J, I
    if not idx_ext and not mid_ext and not rng_ext and pnk_ext:
        if thumb_out:
            return "Y"
        if abs(lm[20].x - lm[17].x) / palm > 0.30:
            return "J"
        return "I"

    # 8. Hooked Index Finger: X (PIP joint 6 raised high, tip 8 bent down, other fingers tight)
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        if dist(lm[6], lm[0]) / palm > 1.15 and lm[8].y > lm[6].y and dist(lm[12], lm[0]) / palm < 1.05:
            return "X"

    # 9. Curved Open Hand: O vs C
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        if 0.14 <= pinch_index <= 0.34 and pinch_mid <= 0.38 and dist(lm[8], lm[0]) / palm > 1.05:
            return "O"
        if 0.36 < pinch_index < 0.85 and dist(lm[8], lm[0]) / palm > 1.10:
            return "C"

    # 10. Closed Fist Variants: A, T, M, N, E, S
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        # A: Thumb upright on the outer side of index knuckle
        if thumb_to_pinky_mcp > 0.72 and lm[4].y < lm[6].y:
            return "A"
        # T: Thumb tip tucked right between index PIP (6) and middle PIP (10)
        if dist(lm[4], lm[6]) / palm < 0.24 and dist(lm[4], lm[10]) / palm < 0.28 and lm[4].y < lm[5].y:
            return "T"
        # M: Thumb tucked deep across palm near ring/pinky base (13 or 17)
        if dist(lm[4], lm[13]) / palm < 0.26 or dist(lm[4], lm[17]) / palm < 0.32:
            return "M"
        # N: Thumb tucked under index/middle near middle base (9)
        if dist(lm[4], lm[9]) / palm < 0.25 and lm[4].y > lm[6].y:
            return "N"
        # E: Fingertips curled above thumb tip
        if lm[8].y < lm[4].y - 0.02 and lm[12].y < lm[4].y - 0.02 and lm[16].y < lm[4].y - 0.02:
            return "E"
        # S: Tight closed fist with thumb wrapped across front
        return "S"

    return "Unknown Gesture"

def classify_word_gesture(lm, handedness):
    """Classifies common words and phrases."""
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    if handedness == "Right":
        fingers.append(1 if lm[4].x > lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x < lm[3].x else 0)

    for idx in range(1, 5):
        fingers.append(1 if lm[tip_ids[idx]].y < lm[tip_ids[idx] - 2].y else 0)

    pattern = tuple(fingers)
    palm_size = max(dist(lm[0], lm[9]), 0.001)
    pinch_dist = dist(lm[4], lm[8]) / palm_size
    spread = dist(lm[8], lm[12]) / palm_size

    if pinch_dist < 0.28 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
        return "OK"
    if pattern == (1, 0, 0, 0, 0):
        return "Good" if lm[4].y < lm[5].y - 0.05 else "No"
    if pattern == (0, 1, 1, 0, 0):
        return "Peace" if spread > 0.22 else "Number 2"
    if pattern == (1, 1, 1, 1, 1):
        return "Hello" if spread > 0.18 else "Thank You"
    if pattern == (0, 1, 1, 1, 1):
        return "Bye"
    if pattern == (0, 0, 0, 0, 0):
        return "Yes"
    if pattern == (0, 1, 1, 1, 0):
        return "Water"
    if pattern == (1, 1, 0, 0, 1):
        return "I Love You"
    if pattern == (1, 0, 0, 0, 1):
        return "Call Me"
    if pattern == (1, 1, 0, 0, 0):
        return "Help"

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

                if recognition_mode == "alphabet":
                    gesture_text = classify_alphabet(landmarks, handedness)
                else:
                    gesture_text = classify_word_gesture(landmarks, handedness)

                current_detected_word = gesture_text

                label_prefix = "Letter" if recognition_mode == "alphabet" else current_sign_language
                cv2.rectangle(frame, (12, 12), (380, 62), (9, 13, 22), -1)
                cv2.putText(
                    frame, f"[{label_prefix}] {gesture_text}", (22, 46),
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


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global recognition_mode
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "words")
    if mode in ["words", "alphabet"]:
        recognition_mode = mode
    return jsonify({"status": "ok", "mode": recognition_mode})


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
    return jsonify({
        "prediction": current_detected_word,
        "language": current_sign_language,
        "mode": recognition_mode
    })


@app.route('/api/dictionary', methods=['GET'])
def get_dictionary():
    return jsonify(WORD_DICTIONARY + ALPHABET_DICTIONARY)


@app.route('/api/chat', methods=['POST'])
def chat_assistant():
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()
    msg_lower = user_msg.lower()
    all_signs = WORD_DICTIONARY + ALPHABET_DICTIONARY

    if not user_msg:
        return jsonify({"reply": "Ask me how to sign any word or English letter (A–Z)!", "action": None})

    if any(k in msg_lower for k in ["start camera", "turn on camera"]):
        return jsonify({"reply": "Starting your webcam now!", "action": "start_camera"})

    if any(k in msg_lower for k in ["stop camera", "turn off camera"]):
        return jsonify({"reply": "Stopping the webcam feed.", "action": "stop_camera"})

    if any(k in msg_lower for k in ["alphabet mode", "spelling mode", "letters mode"]):
        return jsonify({"reply": "Switched to **Alphabet (A–Z) Fingerspelling Mode**!", "action": "mode_alphabet"})

    if any(k in msg_lower for k in ["word mode", "words mode", "phrase mode"]):
        return jsonify({"reply": "Switched to **Words & Phrases Mode**!", "action": "mode_words"})

    # Check if user asks how to sign a specific letter (e.g., "how to sign letter M" or "sign R")
    for item in ALPHABET_DICTIONARY:
        letter = item["word"].lower()
        if (f"letter {letter}" in msg_lower or f"sign {letter}" == msg_lower or f"sign {letter}?" in msg_lower or f"do {letter}" in msg_lower):
            return jsonify({
                "reply": f"**Letter {item['word']}** (English Fingerspelling):\n• **Hand Shape:** {item['gesture']}\n• **Instructions:** {item['description']}",
                "action": "insert_word",
                "word": item["word"]
            })

    # Check words
    for item in WORD_DICTIONARY:
        word_lower = item["word"].lower()
        if word_lower in msg_lower and any(w in msg_lower for w in ["how", "sign", "gesture", "show", "do", "what is"]):
            return jsonify({
                "reply": f"**{item['word']}** ({item['category']}):\n• **Hand Shape:** {item['gesture']}\n• **Instructions:** {item['description']}",
                "action": "insert_word",
                "word": item["word"]
            })

    if any(k in msg_lower for k in ["alphabet", "a to z", "letters", "list signs"]):
        return jsonify({
            "reply": "I support all **26 English Alphabet letters (A–Z)** plus **12 common words/phrases**!\n\nToggle **Alphabet (A–Z)** in the detector header to spell names letter-by-letter, or ask *\"How do I sign letter R?\"*.",
            "action": None
        })

    return jsonify({
        "reply": f"You signed/typed: *\"{user_msg}\"*\n\nTry asking **\"How do I sign letter K?\"** or switch to **Alphabet (A–Z)** mode to spell words!",
        "action": None
    })


if __name__ == '__main__':
    print("Server running! Open your browser at: http://127.0.0.1:5000")
    app.run(debug=False, threaded=True, port=5000)