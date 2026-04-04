import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os

# Load hand detection model
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# All labels — 26 letters + 10 words
LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
WORDS = ['YES', 'NO', 'HELLO', 'PLEASE', 'THANKYOU', 'SORRY', 'HELP', 'LOVE', 'GOOD', 'BAD']
ALL_LABELS = LETTERS + WORDS

# Dataset file
dataset_path = "dataset.csv"

# Check existing samples
def count_samples():
    if not os.path.exists(dataset_path):
        return {}
    counts = {}
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            label = row[-1]
            counts[label] = counts.get(label, 0) + 1
    return counts

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

# Create dataset file if needed
if not os.path.exists(dataset_path):
    with open(dataset_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header.extend([f'x{i}', f'y{i}'])
        header.append('label')
        writer.writerow(header)

# Find which label to start from
existing = count_samples()
print("\nCurrent dataset status:")
for label in ALL_LABELS:
    count = existing.get(label, 0)
    status = "DONE" if count >= 200 else f"{count}/200"
    print(f"  {label}: {status}")

# Find first incomplete label
start_label = None
for label in ALL_LABELS:
    if existing.get(label, 0) < 200:
        start_label = label
        break

if start_label is None:
    print("\n✅ All labels complete!")
    exit()

current_index = ALL_LABELS.index(start_label)
current_label = ALL_LABELS[current_index]
samples_collected = existing.get(current_label, 0)
samples_needed = 200
collecting = False

cap = cv2.VideoCapture(0)

print(f"\nStarting from: {current_label}")
print("Press SPACE to start/stop collecting")
print("Press ENTER to move to next label")
print("Press Q to quit\n")

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

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
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            # Draw skeleton
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
            for point in points:
                cv2.circle(frame, point, 5, (0, 0, 255), -1)

            # Save data if collecting
            if collecting and samples_collected < samples_needed:
                normalized = normalize(hand_landmarks)
                normalized.append(current_label)
                with open(dataset_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(normalized)
                samples_collected += 1

    # Auto stop when done
    if samples_collected >= samples_needed:
        collecting = False

    # Display info on screen
    is_word = current_label in WORDS
    label_type = "WORD" if is_word else "LETTER"
    status = "COLLECTING" if collecting else "PAUSED"
    color = (0, 255, 0) if collecting else (0, 0, 255)
    done_color = (0, 255, 255)

    cv2.putText(frame, f"{label_type}: {current_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Samples: {samples_collected}/{samples_needed}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if samples_collected >= samples_needed:
        cv2.putText(frame, "DONE! Press ENTER for next", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, done_color, 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):
        collecting = not collecting
        print(f"Collecting: {collecting}")
    elif key == 13:  # enter
        if current_index < len(ALL_LABELS) - 1:
            current_index += 1
            current_label = ALL_LABELS[current_index]
            samples_collected = existing.get(current_label, 0)
            collecting = False
            print(f"Moving to: {current_label}")
        else:
            print("All labels complete!")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")