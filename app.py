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

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
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
current_detected_char = "Waiting for gesture..."
recognition_mode = "alphabet"  # "alphabet" or "numbers"
current_accent = "en-IN"
last_timestamp_ms = 0

# Reference Dictionary: ONLY A-Z Alphabets, 0-9 Numbers, and SPACE
REFERENCE_DICTIONARY = [
    # Control
    {"symbol": "SPACE", "hands": "2 Hands", "category": "Control", "gesture": "Both palms pressed together upright OR both open hands side-by-side", "description": "Inserts a space between spelled words."},

    # Numbers 0-9
    {"symbol": "0", "hands": "1 Hand", "category": "Numbers", "gesture": "All fingers curled into a closed fist (0 fingers up)", "description": "Close all fingers into a fist in Numbers mode."},
    {"symbol": "1", "hands": "1 Hand", "category": "Numbers", "gesture": "1 finger extended (Index straight up)", "description": "Raise 1 finger in Numbers mode."},
    {"symbol": "2", "hands": "1 Hand", "category": "Numbers", "gesture": "2 fingers extended (Index + Middle)", "description": "Raise 2 fingers in Numbers mode."},
    {"symbol": "3", "hands": "1 Hand", "category": "Numbers", "gesture": "3 fingers extended (Thumb/Index/Middle or 3 fingers)", "description": "Raise 3 fingers in Numbers mode."},
    {"symbol": "4", "hands": "1 Hand", "category": "Numbers", "gesture": "4 fingers extended (Thumb tucked)", "description": "Raise 4 fingers in Numbers mode."},
    {"symbol": "5", "hands": "1 Hand", "category": "Numbers", "gesture": "All 5 fingers extended on one hand", "description": "Raise all 5 fingers on one hand in Numbers mode."},
    {"symbol": "6", "hands": "2 Hands", "category": "Numbers", "gesture": "5 fingers on one hand + 1 finger on second hand (6 total)", "description": "Show 5 fingers on one hand and 1 finger on the other."},
    {"symbol": "7", "hands": "2 Hands", "category": "Numbers", "gesture": "5 fingers on one hand + 2 fingers on second hand (7 total)", "description": "Show 5 fingers on one hand and 2 fingers on the other."},
    {"symbol": "8", "hands": "2 Hands", "category": "Numbers", "gesture": "5 fingers on one hand + 3 fingers on second hand (8 total)", "description": "Show 5 fingers on one hand and 3 fingers on the other."},
    {"symbol": "9", "hands": "2 Hands", "category": "Numbers", "gesture": "5 fingers on one hand + 4 fingers on second hand (9 total)", "description": "Show 5 fingers on one hand and 4 fingers on the other."},

    # Alphabets A-Z (Two-Hand Priority + Single-Hand Support)
    {"symbol": "A", "hands": "2 Hands", "category": "Alphabet", "gesture": "Point index finger to other open hand's THUMB tip", "description": "Hold one hand open and touch its thumb tip with your other index finger."},
    {"symbol": "B", "hands": "2 Hands (or 1)", "category": "Alphabet", "gesture": "2-Hand: Both hands form thumb-index circles touching | 1-Hand: 4 fingers up", "description": "Form a ring with thumb and index on each hand and press the two rings together."},
    {"symbol": "C", "hands": "1 Hand", "category": "Alphabet", "gesture": "Thumb and fingers curved into a C arc", "description": "Curve your thumb and fingers into a clear C shape."},
    {"symbol": "D", "hands": "2 Hands (or 1)", "category": "Alphabet", "gesture": "2-Hand: Index & thumb curve onto other upright index | 1-Hand: Index straight up", "description": "Hold one index finger up and touch its top and base with your other thumb and index."},
    {"symbol": "E", "hands": "2 Hands", "category": "Alphabet", "gesture": "Point index finger to other open hand's INDEX fingertip", "description": "Hold one hand open and touch its index fingertip with your other index finger."},
    {"symbol": "F", "hands": "1 Hand", "category": "Alphabet", "gesture": "Thumb & index pinched in a circle with 3 fingers raised", "description": "Touch thumb and index tips together while raising middle, ring, and pinky."},
    {"symbol": "G", "hands": "2 Hands (or 1)", "category": "Alphabet", "gesture": "2-Hand: Two closed fists stacked vertically | 1-Hand: Index & thumb horizontal", "description": "Stack one closed fist on top of your other closed fist."},
    {"symbol": "H", "hands": "1 Hand", "category": "Alphabet", "gesture": "Index and middle fingers pointing horizontally together", "description": "Point your index and middle fingers horizontally to the side."},
    {"symbol": "I", "hands": "2 Hands (or 1)", "category": "Alphabet", "gesture": "2-Hand: Point index to other hand's MIDDLE fingertip | 1-Hand: Pinky up", "description": "Hold one hand open and touch its middle fingertip with your other index finger."},
    {"symbol": "J", "hands": "1 Hand", "category": "Alphabet", "gesture": "Pinky extended and tilted diagonally inward", "description": "Extend only your pinky finger and tilt your wrist inward."},
    {"symbol": "K", "hands": "2 Hands (or 1)", "category": "Alphabet", "gesture": "2-Hand: Index touches middle joint of other upright index | 1-Hand: V + thumb", "description": "Hold one index finger upright and touch its middle joint with your other index finger."},
    {"symbol": "L", "hands": "1 Hand", "category": "Alphabet", "gesture": "Thumb and index finger extended in a 90° L-shape", "description": "Extend your index finger straight up and thumb straight out."},
    {"symbol": "M", "hands": "2 Hands", "category": "Alphabet", "gesture": "Press 3 fingers (index, middle, ring) flat onto other open palm", "description": "Hold one palm open and place 3 fingers of your other hand flat on the palm."},
    {"symbol": "N", "hands": "2 Hands", "category": "Alphabet", "gesture": "Press 2 fingers (index, middle) flat onto other open palm", "description": "Hold one palm open and place 2 fingers of your other hand flat on the palm."},
    {"symbol": "O", "hands": "2 Hands (or 1)", "category": "Alphabet", "gesture": "2-Hand: Point index to other hand's RING fingertip | 1-Hand: Fingers in O circle", "description": "Hold one hand open and touch its ring fingertip with your other index finger."},
    {"symbol": "P", "hands": "1 Hand", "category": "Alphabet", "gesture": "Index and middle fingers pointing straight downward", "description": "Point your index and middle fingers downward toward the floor."},
    {"symbol": "Q", "hands": "1 Hand", "category": "Alphabet", "gesture": "Only index finger pointing straight downward", "description": "Point your index finger downward toward the floor."},
    {"symbol": "R", "hands": "1 Hand", "category": "Alphabet", "gesture": "Index and middle fingers crossed over each other", "description": "Raise index and middle fingers and cross them over each other."},
    {"symbol": "S", "hands": "2 Hands (or 1)", "category": "Alphabet", "gesture": "2-Hand: Both pinky fingers hooked/touching together | 1-Hand: Closed fist", "description": "Extend both pinky fingers and hook/touch them together."},
    {"symbol": "T", "hands": "2 Hands", "category": "Alphabet", "gesture": "Point index finger into the center of other open palm (T-shape)", "description": "Hold one palm open and point your other index finger into the middle of the palm."},
    {"symbol": "U", "hands": "2 Hands (or 1)", "category": "Alphabet", "gesture": "2-Hand: Point index to other hand's PINKY fingertip | 1-Hand: 2 fingers together", "description": "Hold one hand open and touch its pinky fingertip with your other index finger."},
    {"symbol": "V", "hands": "1 Hand", "category": "Alphabet", "gesture": "Index and middle fingers raised and spread apart in a V", "description": "Raise index and middle fingers apart in a V."},
    {"symbol": "W", "hands": "2 Hands (or 1)", "category": "Alphabet", "gesture": "2-Hand: Both open hands overlapped at fingers | 1-Hand: 3 fingers up", "description": "Overlap/interlock the open fingers of both hands together."},
    {"symbol": "X", "hands": "2 Hands", "category": "Alphabet", "gesture": "Both index fingers crossed over each other in an X", "description": "Extend both index fingers and cross them in an X shape."},
    {"symbol": "Y", "hands": "1 Hand", "category": "Alphabet", "gesture": "Thumb and pinky extended outward (hang-loose shape)", "description": "Extend thumb and pinky outward while folding middle three fingers."},
    {"symbol": "Z", "hands": "1 Hand", "category": "Alphabet", "gesture": "Index finger pointing diagonally across", "description": "Extend only your index finger diagonally across."}
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


def draw_hand_skeleton(frame, landmarks, hand_idx=0):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    line_color = (56, 189, 248) if hand_idx == 0 else (245, 158, 11)
    dot_color = (16, 185, 129) if hand_idx == 0 else (168, 85, 247)
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], line_color, 2)
    for px, py in points:
        cv2.circle(frame, (px, py), 5, dot_color, -1)


def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def get_finger_states(lm):
    """Mirror-invariant finger extension detector."""
    palm = max(dist(lm[0], lm[9]), 0.001)
    idx = dist(lm[8], lm[0]) / palm > 1.35 and lm[8].y < lm[6].y
    mid = dist(lm[12], lm[0]) / palm > 1.35 and lm[12].y < lm[10].y
    rng = dist(lm[16], lm[0]) / palm > 1.30 and lm[16].y < lm[14].y
    pnk = dist(lm[20], lm[0]) / palm > 1.20 and lm[20].y < lm[18].y
    thm = dist(lm[4], lm[17]) / palm > 0.85 and dist(lm[4], lm[5]) / palm > 0.42
    count = int(thm) + int(idx) + int(mid) + int(rng) + int(pnk)
    return {"thm": thm, "idx": idx, "mid": mid, "rng": rng, "pnk": pnk, "count": count, "palm": palm}


def check_two_hand_space(lm1, lm2):
    """Detects two-hand SPACE gesture (both palms pressed together or open hands side-by-side)."""
    s1 = get_finger_states(lm1)
    s2 = get_finger_states(lm2)
    avg_palm = (s1["palm"] + s2["palm"]) / 2.0

    h1_open = s1["idx"] and s1["mid"] and s1["rng"] and s1["pnk"]
    h2_open = s2["idx"] and s2["mid"] and s2["rng"] and s2["pnk"]

    if h1_open and h2_open:
        wrist_dist = dist(lm1[0], lm2[0]) / avg_palm
        mid_tip_dist = dist(lm1[12], lm2[12]) / avg_palm
        pinky_dist = dist(lm1[20], lm2[20]) / avg_palm
        # Palms pressed together OR open hands side-by-side (book pose)
        if (wrist_dist < 0.80 and mid_tip_dist < 0.70) or (pinky_dist < 0.55 and wrist_dist < 1.15):
            return True
    return False


def classify_numbers_mode(hands):
    """Pure 0-9 number classifier across 1 or 2 hands + SPACE."""
    if len(hands) >= 2 and check_two_hand_space(hands[0], hands[1]):
        return "SPACE"

    total_fingers = sum(get_finger_states(h)["count"] for h in hands)
    if 0 <= total_fingers <= 9:
        return str(total_fingers)
    return "Unknown"


def classify_two_hand_alphabet(lm1, lm2):
    """High-accuracy dual-hand A-Z classifier (no word collisions)."""
    if check_two_hand_space(lm1, lm2):
        return "SPACE"

    s1 = get_finger_states(lm1)
    s2 = get_finger_states(lm2)
    avg_palm = (s1["palm"] + s2["palm"]) / 2.0

    for ptr, base, sp, sb in [(lm1, lm2, s1, s2), (lm2, lm1, s2, s1)]:
        p_idx_only = sp["idx"] and not sp["mid"] and not sp["rng"] and not sp["pnk"]
        b_open_all = sb["idx"] and sb["mid"] and sb["rng"] and sb["pnk"]

        # X: Both index fingers extended and crossed/touching at middle joints
        if p_idx_only and sb["idx"] and not sb["mid"] and not sb["rng"] and not sb["pnk"]:
            if dist(ptr[6], base[6]) / avg_palm < 0.45:
                return "X"
            if dist(ptr[8], base[8]) / avg_palm < 0.36 and dist(ptr[4], base[5]) / avg_palm < 0.48:
                return "D"
            if dist(ptr[8], base[6]) / avg_palm < 0.34:
                return "K"

        # Two-Hand Vowels (A, E, I, O, U) + T (Center of Palm)
        if p_idx_only and b_open_all:
            if dist(ptr[8], base[4]) / avg_palm < 0.34:
                return "A"
            if dist(ptr[8], base[8]) / avg_palm < 0.32:
                return "E"
            if dist(ptr[8], base[12]) / avg_palm < 0.32:
                return "I"
            if dist(ptr[8], base[16]) / avg_palm < 0.32:
                return "O"
            if dist(ptr[8], base[20]) / avg_palm < 0.32:
                return "U"
            if dist(ptr[8], base[9]) / avg_palm < 0.38:
                return "T"

        # M (3 fingers on open palm) & N (2 fingers on open palm)
        if b_open_all and dist(ptr[8], base[9]) / avg_palm < 0.48:
            if sp["idx"] and sp["mid"] and sp["rng"] and not sp["pnk"]:
                return "M"
            if sp["idx"] and sp["mid"] and not sp["rng"] and not sp["pnk"]:
                return "N"

        # B: Both hands making thumb-index circles touching
        p_ring = dist(ptr[4], ptr[8]) / sp["palm"] < 0.36
        b_ring = dist(base[4], base[8]) / sb["palm"] < 0.36
        if p_ring and b_ring and dist(ptr[8], base[8]) / avg_palm < 0.40:
            return "B"

        # G: Two closed fists stacked vertically
        p_fist = not (sp["idx"] or sp["mid"] or sp["rng"] or sp["pnk"])
        b_fist = not (sb["idx"] or sb["mid"] or sb["rng"] or sb["pnk"])
        if p_fist and b_fist and dist(ptr[9], base[9]) / avg_palm < 0.68:
            return "G"

        # S: Both pinkies extended and hooked/touching
        if sp["pnk"] and not sp["idx"] and sb["pnk"] and not sb["idx"]:
            if dist(ptr[20], base[20]) / avg_palm < 0.38:
                return "S"

        # W: Both open hands overlapped at middle knuckles
        if b_open_all and sp["idx"] and sp["mid"] and sp["rng"] and sp["pnk"]:
            if dist(ptr[10], base[10]) / avg_palm < 0.42:
                return "W"

    return None


def classify_single_alphabet(lm):
    """Single-hand fallback A-Z classifier."""
    palm = max(dist(lm[0], lm[9]), 0.001)
    idx_ext = dist(lm[8], lm[0]) / palm > 1.35 and lm[8].y < lm[6].y
    mid_ext = dist(lm[12], lm[0]) / palm > 1.35 and lm[12].y < lm[10].y
    rng_ext = dist(lm[16], lm[0]) / palm > 1.30 and lm[16].y < lm[14].y
    pnk_ext = dist(lm[20], lm[0]) / palm > 1.20 and lm[20].y < lm[18].y

    idx_down = lm[8].y > lm[5].y + (0.45 * palm)
    mid_down = lm[12].y > lm[9].y + (0.45 * palm)
    idx_horiz = abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y) * 1.25 and dist(lm[8], lm[5]) / palm > 0.50
    mid_horiz = abs(lm[12].x - lm[9].x) > abs(lm[12].y - lm[9].y) * 1.25 and dist(lm[12], lm[9]) / palm > 0.50

    thumb_to_pinky_mcp = dist(lm[4], lm[17]) / palm
    thumb_to_index_mcp = dist(lm[4], lm[5]) / palm
    thumb_out = thumb_to_pinky_mcp > 0.85 and thumb_to_index_mcp > 0.42
    pinch_index = dist(lm[4], lm[8]) / palm
    idx_mid_spread = dist(lm[8], lm[12]) / palm

    if idx_down and mid_down and not rng_ext and not pnk_ext:
        return "P"
    if idx_down and not mid_down and not rng_ext and not pnk_ext:
        return "Q"
    if idx_horiz and mid_horiz and not rng_ext and not pnk_ext:
        return "H"
    if idx_horiz and not mid_horiz and not rng_ext and not pnk_ext:
        return "G"
    if idx_ext and mid_ext and rng_ext and pnk_ext:
        return "B"
    if not idx_ext and mid_ext and rng_ext and pnk_ext:
        return "F"
    if idx_ext and mid_ext and rng_ext and not pnk_ext:
        return "W"
    if idx_ext and mid_ext and not rng_ext and not pnk_ext:
        if dist(lm[8], lm[13]) < dist(lm[12], lm[13]) - 0.015:
            return "R"
        if lm[4].y < lm[6].y and idx_mid_spread > 0.20:
            return "K"
        return "V" if idx_mid_spread > 0.24 else "U"
    if idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        if thumb_out:
            return "L"
        if abs(lm[8].x - lm[5].x) / palm > 0.38:
            return "Z"
        return "D"
    if not idx_ext and not mid_ext and not rng_ext and pnk_ext:
        if thumb_out:
            return "Y"
        if abs(lm[20].x - lm[17].x) / palm > 0.30:
            return "J"
        return "I"
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        if 0.14 <= pinch_index <= 0.34 and dist(lm[8], lm[0]) / palm > 1.05:
            return "O"
        if 0.36 < pinch_index < 0.85 and dist(lm[8], lm[0]) / palm > 1.10:
            return "C"
        if thumb_to_pinky_mcp > 0.72 and lm[4].y < lm[6].y:
            return "A"
        return "S"

    return "Unknown"


def generate_frames():
    global camera, is_camera_active, current_detected_char, last_timestamp_ms

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
        hands = detection_result.hand_landmarks

        if hands:
            for idx, landmarks in enumerate(hands):
                draw_hand_skeleton(frame, landmarks, hand_idx=idx)

            hand_count = len(hands)
            detected = None

            if recognition_mode == "numbers":
                detected = classify_numbers_mode(hands)
            else:
                if hand_count >= 2:
                    detected = classify_two_hand_alphabet(hands[0], hands[1])
                if not detected:
                    detected = classify_single_alphabet(hands[0])

            current_detected_char = detected
            mode_tag = "0-9" if recognition_mode == "numbers" else "A-Z"

            cv2.rectangle(frame, (12, 12), (390, 62), (9, 13, 22), -1)
            cv2.putText(
                frame, f"[{hand_count}H | {mode_tag}] {detected}", (22, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (56, 189, 248), 2, cv2.LINE_AA
            )
        else:
            current_detected_char = "No hand detected"

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
    global camera, is_camera_active, current_detected_char
    if is_camera_active and camera is not None and camera.isOpened():
        return jsonify({"status": "started"})

    cam = open_working_camera()
    if cam is None:
        is_camera_active = False
        current_detected_char = "Camera not found"
        return jsonify({
            "status": "error",
            "message": "Could not access webcam. Close other camera apps or check Camera permissions."
        }), 500

    camera = cam
    is_camera_active = True
    current_detected_char = "Waiting for gesture..."
    return jsonify({"status": "started"})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, is_camera_active, current_detected_char
    is_camera_active = False
    time.sleep(0.1)
    if camera is not None:
        camera.release()
        camera = None
    current_detected_char = "Camera stopped"
    return jsonify({"status": "stopped"})


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global recognition_mode
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "alphabet")
    if mode in ["alphabet", "numbers"]:
        recognition_mode = mode
    return jsonify({"status": "ok", "mode": recognition_mode})


@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    return jsonify({
        "prediction": current_detected_char,
        "mode": recognition_mode
    })


@app.route('/api/dictionary', methods=['GET'])
def get_dictionary():
    return jsonify(REFERENCE_DICTIONARY)


@app.route('/api/chat', methods=['POST'])
def chat_assistant():
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()
    msg_lower = user_msg.lower()

    if not user_msg:
        return jsonify({"reply": "Ask how to sign any letter (A–Z), number (0–9), or SPACE!", "action": None})

    if any(k in msg_lower for k in ["start camera", "turn on camera"]):
        return jsonify({"reply": "Starting dual-hand camera tracking!", "action": "start_camera"})

    if any(k in msg_lower for k in ["stop camera", "turn off camera"]):
        return jsonify({"reply": "Stopping camera feed.", "action": "stop_camera"})

    if any(k in msg_lower for k in ["number mode", "numbers mode", "0-9"]):
        return jsonify({"reply": "Switched to **Numbers (0–9) Mode**! Show 0–5 fingers on one hand, or 6–9 fingers across two hands.", "action": "mode_numbers"})

    if any(k in msg_lower for k in ["alphabet mode", "a-z", "letters mode"]):
        return jsonify({"reply": "Switched to **Alphabet (A–Z) Mode**!", "action": "mode_alphabet"})

    if "space" in msg_lower:
        return jsonify({
            "reply": "**SPACE (2 Hands):** Press both palms together upright (or hold both open palms side-by-side) for 1.5s to insert a space!",
            "action": "insert_char",
            "symbol": "SPACE"
        })

    for item in REFERENCE_DICTIONARY:
        sym = item["symbol"].lower()
        if f"letter {sym}" in msg_lower or f"number {sym}" in msg_lower or f"sign {sym}" == msg_lower or f"sign {sym}?" in msg_lower:
            return jsonify({
                "reply": f"**{item['symbol']}** ({item['category']} | {item['hands']}):\n• **Gesture:** {item['gesture']}\n• **Instructions:** {item['description']}",
                "action": "insert_char",
                "symbol": item["symbol"]
            })

    return jsonify({
        "reply": "**Dual-Hand Vowels (A, E, I, O, U):**\nHold one hand open and touch its fingertips with your other index finger:\n• **Thumb** = `A` | **Index** = `E` | **Middle** = `I` | **Ring** = `O` | **Pinky** = `U`\n• **Both Palms Together** = `SPACE`",
        "action": None
    })


if __name__ == '__main__':
    print("Server running! Open your browser at: http://127.0.0.1:5000")
    app.run(debug=False, threaded=True, port=5000)