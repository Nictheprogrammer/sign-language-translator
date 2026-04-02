import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
import time

# Load hand detection model
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Create dataset file if it doesn't exist
dataset_path = "dataset.csv"
if not os.path.exists(dataset_path):
    with open(dataset_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header row: x0,y0,x1,y1...x20,y20,label
        header = []
        for i in range(21):
            header.extend([f'x{i}', f'y{i}'])
        header.append('label')
        writer.writerow(header)

def normalize(landmarks):
    # Get wrist position (landmark 0)
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y

    # Subtract wrist from all landmarks
    normalized = []
    for lm in landmarks:
        normalized.extend([
            round(lm.x - wrist_x, 4),
            round(lm.y - wrist_y, 4)
        ])
    return normalized

cap = cv2.VideoCapture(0)

current_letter = 'A'
samples_collected = 0
samples_needed = 200
collecting = False

print(f"Ready to collect data for letter: {current_letter}")
print("Press SPACE to start/stop collecting")
print("Press ENTER to move to next letter")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Draw dots
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for point in points:
                cv2.circle(frame, point, 5, (0, 0, 255), -1)

            # Save data if collecting
            if collecting and samples_collected < samples_needed:
                normalized = normalize(hand_landmarks)
                normalized.append(current_letter)
                with open(dataset_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(normalized)
                samples_collected += 1

    # Show status on screen
    status = "COLLECTING" if collecting else "PAUSED"
    color = (0, 255, 0) if collecting else (0, 0, 255)
    cv2.putText(frame, f"Letter: {current_letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Samples: {samples_collected}/{samples_needed}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Auto stop when enough samples collected
    if samples_collected >= samples_needed:
        collecting = False
        cv2.putText(frame, "DONE! Press ENTER for next letter", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):  # spacebar
        collecting = not collecting
        print(f"Collecting: {collecting}")
    elif key == 13:  # enter key
        if current_letter < 'Z':
            current_letter = chr(ord(current_letter) + 1)
            samples_collected = 0
            collecting = False
            print(f"Moving to letter: {current_letter}")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")