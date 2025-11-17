import cv2
import time

print("Testing camera window...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not available")
    exit()

print("✅ Camera opened - Window should appear!")
cv2.namedWindow("Camera Test", cv2.WINDOW_AUTOSIZE)

# Show a simple camera feed for 10 seconds
start_time = time.time()
while time.time() - start_time < 10:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Camera Test - Press ESC to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {int(10 - (time.time() - start_time))}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("Camera Test", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
    else:
        print("❌ Cannot read from camera")
        break

cap.release()
cv2.destroyAllWindows()
print("Camera test completed!")