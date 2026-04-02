import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import collections

# Load the trained model
print("Loading model...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded!")

# Load hand detection model
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def normalize(landmarks):
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    normalized = []
    for lm in landmarks:
        normalized.extend([
            round(lm.x - wrist_x, 4),
            round(lm.y - wrist_y, 4)
        ])
    return normalized

# Smoothing buffer — stores last 10 predictions
# instead of showing every single prediction instantly
# which would flicker too much
buffer = collections.deque(maxlen=10)

cap = cv2.VideoCapture(0)
print("Starting prediction... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    predicted_letter = None

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Draw skeleton
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
            for point in points:
                cv2.circle(frame, point, 5, (0, 0, 255), -1)

            # Normalize and predict
            features = normalize(hand_landmarks)
            prediction = model.predict([features])[0]
            buffer.append(prediction)

            # Only show prediction if last 10 frames agree
            if len(buffer) == 10 and len(set(buffer)) == 1:
                predicted_letter = prediction

    # Display prediction on screen
    if predicted_letter:
        cv2.putText(frame, predicted_letter, (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 10)
    else:
        cv2.putText(frame, "...", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (100, 100, 100), 10)

    cv2.imshow("Sign Language Predictor", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()