import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Hand connection pairs for drawing lines between dots
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# Download the hand detection model if you don't have it yet
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand model... (only happens once)")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Download done!")

# Load the hand detection model
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hand
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Convert landmark positions to pixel coordinates
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            # Print raw landmark coordinates
            print([(round(lm.x, 2), round(lm.y, 2)) for lm in hand_landmarks])

            # Draw lines between dots
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

            # Draw dots
            for point in points:
                cv2.circle(frame, point, 5, (0, 0, 255), -1)

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()