import cv2
import mediapipe as mp
from collections import deque
import threading
import queue
import time

# === Settings ===
VIDEO_PATHS = {
    "tongue_shake": "Assets/orca.mp4",
    "thumbs_up": "Assets/thumbsup.gif",
    "eyebrow_raise": "Assets/eyebrow_raise.mp4"
}
SHAKE_WINDOW = 15
SHAKE_THRESHOLD = 0.03
TONGUE_THRESHOLD = 0.01
MIN_MOUTH_OPEN = 0.01
EYEBROW_THRESHOLD = 0.02
THUMBS_UP_CONFIDENCE = 0.7
TRIGGER_COOLDOWN = 60
SUSTAIN_FRAMES = 7

# === Setup MediaPipe ===
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# === Setup Camera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()
else:
    print("✅ Camera opened successfully")

nose_positions = deque(maxlen=SHAKE_WINDOW)
eyebrow_positions = deque(maxlen=SHAKE_WINDOW)
gesture_frames = 0
cooldown = 0
current_gesture = None

# === Queue and flag for video frames ===
video_queue = queue.Queue(maxsize=1)
play_video_flag = threading.Event()
video_window_open = False

# === Function to read video frames in a separate thread ===
def video_reader():
    global current_gesture
    while True:
        play_video_flag.wait()  # wait until flagged
        if current_gesture and current_gesture in VIDEO_PATHS:
            video_path = VIDEO_PATHS[current_gesture]
            print(f"🎬 Loading {current_gesture} media: {video_path}")
            vid = cv2.VideoCapture(video_path)
            if not vid.isOpened():
                print("❌ Could not open media:", video_path)
                play_video_flag.clear()
                continue

            print("✅ Media loaded successfully!")
            while vid.isOpened() and play_video_flag.is_set():
                ret, frame = vid.read()
                if not ret:
                    print("🎬 Media finished")
                    break
                if not video_queue.full():
                    video_queue.put(frame)
                time.sleep(0.01)  # reduce CPU usage

            vid.release()
            play_video_flag.clear()  # done playing
            current_gesture = None

# Start video reader thread
threading.Thread(target=video_reader, daemon=True).start()

# === Gesture detection functions ===
def detect_tongue(landmarks):
    upper_lip = landmarks[13].y
    lower_lip = landmarks[14].y
    tongue_tip = landmarks[16].y
    mouth_height = lower_lip - upper_lip
    if mouth_height < MIN_MOUTH_OPEN:
        return False
    return (tongue_tip - lower_lip) > TONGUE_THRESHOLD

def detect_head_shake():
    if len(nose_positions) < SHAKE_WINDOW:
        return False
    motion = max(nose_positions) - min(nose_positions)
    return motion > SHAKE_THRESHOLD

def detect_eyebrow_raise(landmarks):
    # Get eyebrow and eye landmarks
    left_eyebrow = landmarks[70].y  # Left eyebrow
    right_eyebrow = landmarks[295].y  # Right eyebrow
    left_eye = landmarks[159].y  # Left eye
    right_eye = landmarks[386].y  # Right eye
    
    # Calculate average eyebrow-to-eye distance
    avg_distance = ((left_eye - left_eyebrow) + (right_eye - right_eyebrow)) / 2
    eyebrow_positions.append(avg_distance)
    
    if len(eyebrow_positions) < SHAKE_WINDOW:
        return False
    
    # Check if eyebrows are raised (increased distance)
    current_avg = sum(list(eyebrow_positions)[-5:]) / 5  # Last 5 frames
    baseline_avg = sum(list(eyebrow_positions)[:5]) / 5  # First 5 frames
    
    return (current_avg - baseline_avg) > EYEBROW_THRESHOLD

def detect_thumbs_up(hand_landmarks):
    if not hand_landmarks:
        return False
    
    for hand in hand_landmarks:
        landmarks = hand.landmark
        
        # Thumb tip and other fingertips
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Wrist position
        wrist = landmarks[0]
        
        # Check if thumb is extended upward
        thumb_up = thumb_tip.y < wrist.y
        
        # Check if other fingers are folded (tips below MCP joints)
        index_folded = index_tip.y > landmarks[5].y
        middle_folded = middle_tip.y > landmarks[9].y
        ring_folded = ring_tip.y > landmarks[13].y
        pinky_folded = pinky_tip.y > landmarks[17].y
        
        if thumb_up and index_folded and middle_folded and ring_folded and pinky_folded:
            return True
    
    return False

# === Main loop ===
print("🎥 Starting camera... Press ESC to quit")
print("👅 Stick out your tongue and shake your head to trigger video!")
print("📺 Camera window should appear now...")

# Create window
cv2.namedWindow("Facial Gesture Detection", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Facial Gesture Detection", 100, 100)  # Move to specific position

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read from camera")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Add text overlay to show the window is working
    cv2.putText(frame, "FreakDetector - Press ESC to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Gestures: Tongue+Shake | Eyebrow Raise | Thumbs Up", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Process face and hand landmarks
    face_result = face_mesh.process(frame_rgb)
    hands_result = hands.process(frame_rgb)
    gesture_detected = False
    detected_gesture = None

    # Check face gestures
    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0].landmark
        nose_positions.append(face_landmarks[1].x)

        tongue_out = detect_tongue(face_landmarks)
        head_shake = detect_head_shake()
        eyebrow_raised = detect_eyebrow_raise(face_landmarks)

        if tongue_out and head_shake:
            gesture_detected = True
            detected_gesture = "tongue_shake"
            print(f"👅🤯 Tongue + Shake detected! Frame {gesture_frames + 1}/{SUSTAIN_FRAMES}")
        elif eyebrow_raised:
            gesture_detected = True
            detected_gesture = "eyebrow_raise"
            print(f"🤨 Eyebrow raise detected! Frame {gesture_frames + 1}/{SUSTAIN_FRAMES}")
    
    # Check hand gestures
    if hands_result.multi_hand_landmarks:
        thumbs_up = detect_thumbs_up(hands_result.multi_hand_landmarks)
        if thumbs_up:
            gesture_detected = True
            detected_gesture = "thumbs_up"
            print(f"👍 Thumbs up detected! Frame {gesture_frames + 1}/{SUSTAIN_FRAMES}")

    # Count consecutive frames
    if gesture_detected:
        gesture_frames += 1
        if current_gesture != detected_gesture:
            current_gesture = detected_gesture
    else:
        gesture_frames = 0
        current_gesture = None

    # Trigger video if sustained
    if gesture_frames >= SUSTAIN_FRAMES and cooldown == 0 and current_gesture:
        print(f"Gesture '{current_gesture}' sustained! Starting media...")
        play_video_flag.set()
        cooldown = TRIGGER_COOLDOWN
        gesture_frames = 0

    # Cooldown counter
    if cooldown > 0:
        cooldown -= 1

    # Show webcam frame
    cv2.imshow("Facial Gesture Detection", frame)
    
    # Force window update
    cv2.waitKey(1)

    # Show video if available
    if not video_queue.empty():
        video_frame = video_queue.get()
        cv2.imshow("Video Playback", video_frame)
        video_window_open = True
    elif not play_video_flag.is_set() and video_window_open:
        # Video finished, close window
        cv2.destroyWindow("Video Playback")
        video_window_open = False

    # Check for ESC key press to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
