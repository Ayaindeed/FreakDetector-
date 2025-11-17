import streamlit as st
import cv2
import mediapipe as mp
from collections import deque
import numpy as np
import time
from PIL import Image
import io
import base64
import os

# === App Description ===
st.set_page_config(
    page_title="FreakDetector", 
    page_icon="🎭", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
# FreakDetector - AI Gesture Recognition App

**What does this app do?**

An intelligent gesture recognition system that uses your camera to detect specific gestures and responds with entertaining media content. Using advanced AI (MediaPipe), it tracks your facial expressions and hand movements in real-time.

---
""")

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
EYEBROW_THRESHOLD = 0.015
SUSTAIN_FRAMES = 7

# === Setup MediaPipe ===
@st.cache_resource
def load_mediapipe():
    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
    return face_mesh, hands, mp_face, mp_hands

# === Initialize session state ===
if 'nose_positions' not in st.session_state:
    st.session_state.nose_positions = deque(maxlen=SHAKE_WINDOW)
    st.session_state.eyebrow_positions = deque(maxlen=SHAKE_WINDOW)
    st.session_state.gesture_frames = 0
    st.session_state.current_gesture = None
    st.session_state.gesture_triggered = False
    st.session_state.last_trigger_time = 0

# === Gesture detection functions ===
def detect_tongue(landmarks):
    upper_lip = landmarks[13].y
    lower_lip = landmarks[14].y
    tongue_tip = landmarks[16].y
    mouth_height = lower_lip - upper_lip
    if mouth_height < MIN_MOUTH_OPEN:
        return False
    return (tongue_tip - lower_lip) > TONGUE_THRESHOLD

def detect_head_shake(nose_positions):
    if len(nose_positions) < SHAKE_WINDOW:
        return False
    motion = max(nose_positions) - min(nose_positions)
    return motion > SHAKE_THRESHOLD

def detect_eyebrow_raise(landmarks, eyebrow_positions):
    # Use more reliable eyebrow landmarks
    left_eyebrow_inner = landmarks[70].y   # Left eyebrow inner
    left_eyebrow_outer = landmarks[107].y  # Left eyebrow outer
    right_eyebrow_inner = landmarks[296].y # Right eyebrow inner  
    right_eyebrow_outer = landmarks[334].y # Right eyebrow outer
    
    # Eye landmarks for reference
    left_eye_top = landmarks[159].y
    right_eye_top = landmarks[386].y
    
    # Calculate eyebrow height relative to eyes
    left_brow_height = left_eye_top - ((left_eyebrow_inner + left_eyebrow_outer) / 2)
    right_brow_height = right_eye_top - ((right_eyebrow_inner + right_eyebrow_outer) / 2)
    avg_brow_height = (left_brow_height + right_brow_height) / 2
    
    eyebrow_positions.append(avg_brow_height)
    
    if len(eyebrow_positions) < 10:  # Need more frames for baseline
        return False
    
    # Compare recent frames to baseline
    recent_avg = sum(list(eyebrow_positions)[-3:]) / 3  # Last 3 frames
    baseline_avg = sum(list(eyebrow_positions)[:7]) / 7  # First 7 frames for stable baseline
    
    # Eyebrows are raised if height increased significantly
    threshold_increase = 0.015  # Adjusted threshold
    return (recent_avg - baseline_avg) > threshold_increase

def detect_thumbs_up(hand_landmarks):
    if not hand_landmarks:
        return False
    
    for hand in hand_landmarks:
        landmarks = hand.landmark
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        thumb_up = thumb_tip.y < wrist.y
        index_folded = index_tip.y > landmarks[5].y
        middle_folded = middle_tip.y > landmarks[9].y
        ring_folded = ring_tip.y > landmarks[13].y
        pinky_folded = pinky_tip.y > landmarks[17].y
        
        if thumb_up and index_folded and middle_folded and ring_folded and pinky_folded:
            return True
    
    return False

def get_media_display(video_path):
    """Display video or GIF with proper formatting"""
    try:
        if not os.path.exists(video_path):
            return f"<p style='color: red;'>❌ Media file not found: {video_path}</p>"
            
        with open(video_path, "rb") as f:
            media_bytes = f.read()
        media_base64 = base64.b64encode(media_bytes).decode()
        
        if video_path.endswith('.gif'):
            return f'''
            <div style="text-align: center; padding: 20px;">
                <img src="data:image/gif;base64,{media_base64}" 
                     style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" 
                     autoplay loop>
            </div>
            '''
        else:
            return f'''
            <div style="text-align: center; padding: 20px;">
                <video style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" 
                       controls autoplay loop>
                    <source src="data:video/mp4;base64,{media_base64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            '''
    except Exception as e:
        return f"<p style='color: red;'>❌ Error loading media: {str(e)}</p>"

# === Streamlit App ===
def main():
    # Sidebar controls
    st.sidebar.header("🎮 Controls")
    camera_enabled = st.sidebar.checkbox("📹 Enable Camera", value=True)
    
    if st.sidebar.button("🔄 Reset Detection"):
        # Reset session state
        for key in ['nose_positions', 'eyebrow_positions', 'gesture_frames', 'current_gesture']:
            if key in st.session_state:
                if 'positions' in key:
                    st.session_state[key] = deque(maxlen=SHAKE_WINDOW)
                else:
                    st.session_state[key] = 0 if 'frames' in key else None
    
    # Available gestures
    st.sidebar.header("🎯 Available Gestures")
    st.sidebar.markdown("""
    **👅🤯 Tongue + Head Shake**  
    Stick out tongue while shaking head → Orca video
    
    **🤨 Eyebrow Raise**  
    Raise eyebrows significantly → Eyebrow video
    
    **👍 Thumbs Up**  
    Clear thumbs up gesture → Thumbs up GIF
    """)
    
    # Tips
    st.sidebar.header("💡 Tips")
    st.sidebar.info("""
    • Hold gestures for 2-3 seconds
    • Ensure good lighting
    • Face camera directly
    • Wait for cooldown between triggers
    """)
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Live Camera Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
    with col2:
        st.header("Media Response")
        media_placeholder = st.empty()
        gesture_status = st.empty()
        
        # Show current session stats
        with st.expander("Detection Stats", expanded=False):
            stats_placeholder = st.empty()
    
    if not camera_enabled:
        st.info("Enable camera in the sidebar to start gesture detection!")
        media_placeholder.markdown("""
        <div style="text-align: center; padding: 40px;">
            <h3>Ready to detect gestures!</h3>
            <p>Enable the camera to see the magic happen</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Load MediaPipe
    face_mesh, hands, mp_face, mp_hands = load_mediapipe()
    
    # Camera setup
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("""❌ **Could not access camera!** 
        
        **Troubleshooting steps:**
        1. Check if another app is using your camera
        2. Refresh the page and allow camera permissions
        3. Try a different browser (Chrome/Edge recommended)
        4. Ensure your camera drivers are installed
        """)
        return
    
    # Processing loop
    stframe = video_placeholder.empty()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process landmarks
            face_result = face_mesh.process(frame_rgb)
            hands_result = hands.process(frame_rgb)
            
            gesture_detected = False
            detected_gesture = None
            
            # Face gesture detection
            if face_result.multi_face_landmarks:
                face_landmarks = face_result.multi_face_landmarks[0].landmark
                st.session_state.nose_positions.append(face_landmarks[1].x)
                
                tongue_out = detect_tongue(face_landmarks)
                head_shake = detect_head_shake(st.session_state.nose_positions)
                eyebrow_raised = detect_eyebrow_raise(face_landmarks, st.session_state.eyebrow_positions)
                
                if tongue_out and head_shake:
                    gesture_detected = True
                    detected_gesture = "tongue_shake"
                elif eyebrow_raised:
                    gesture_detected = True
                    detected_gesture = "eyebrow_raise"
            
            # Hand gesture detection
            if hands_result.multi_hand_landmarks:
                thumbs_up = detect_thumbs_up(hands_result.multi_hand_landmarks)
                if thumbs_up:
                    gesture_detected = True
                    detected_gesture = "thumbs_up"
            
            # Update gesture tracking
            if gesture_detected:
                st.session_state.gesture_frames += 1
                st.session_state.current_gesture = detected_gesture
            else:
                st.session_state.gesture_frames = 0
                st.session_state.current_gesture = None
            
            # Trigger media if gesture sustained
            current_time = time.time()
            if (st.session_state.gesture_frames >= SUSTAIN_FRAMES and 
                st.session_state.current_gesture and
                current_time - st.session_state.last_trigger_time > 3):  # 3 second cooldown
                
                st.session_state.gesture_triggered = True
                st.session_state.last_trigger_time = current_time
                
                # Show media response
                if st.session_state.current_gesture in VIDEO_PATHS:
                    video_path = VIDEO_PATHS[st.session_state.current_gesture]
                    if os.path.exists(video_path):
                        media_html = get_media_display(video_path)
                        media_placeholder.markdown(media_html, unsafe_allow_html=True)
                        
                        # Clear previous status and show success
                        gesture_status.success(f"🎉 {st.session_state.current_gesture.replace('_', ' ').title()} triggered!")
                    else:
                        media_placeholder.error(f"❌ Media file not found: {video_path}")
                else:
                    media_placeholder.warning("⚠️ Unknown gesture detected")
            
            # Draw status on frame
            cv2.putText(frame, "FreakDetector Active", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if st.session_state.current_gesture:
                cv2.putText(frame, f"Detecting: {st.session_state.current_gesture}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Frames: {st.session_state.gesture_frames}/{SUSTAIN_FRAMES}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert and display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Status update
            if st.session_state.current_gesture:
                status_placeholder.info(f"🎯 Detecting: {st.session_state.current_gesture.replace('_', ' ').title()} "
                                      f"({st.session_state.gesture_frames}/{SUSTAIN_FRAMES} frames)")
            else:
                status_placeholder.info("👀 Waiting for gesture...")
                
            # Update stats
            stats_placeholder.markdown(f"""
            **Session Statistics:**
            - Frames processed: {len(st.session_state.nose_positions)}
            - Current gesture: {st.session_state.current_gesture or 'None'}
            - Consecutive frames: {st.session_state.gesture_frames}
            - Last trigger: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_trigger_time)) if st.session_state.last_trigger_time > 0 else 'Never'}
            """)
            
            time.sleep(0.1)  # Small delay to prevent overwhelming
            
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

if __name__ == "__main__":
    main()