from flask import Flask, render_template, request, jsonify
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import base64
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model and scaler
print("Loading model...")
with open('model.pkl', 'rb') as f:
    saved = pickle.load(f)
    model = saved['model']
    scaler = saved['scaler']
print("Model loaded!")

# Load hand detection model
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from browser
        data = request.json['image']

        # Remove the header from base64 string
        # Browser sends: "data:image/jpeg;base64,/9j/4AAQ..."
        # We only want:  "/9j/4AAQ..."
        image_data = base64.b64decode(data.split(',')[1])

        # Convert to numpy array then to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Normalize landmarks
                features = normalize(hand_landmarks)

                # Scale features the same way we scaled training data
                features_scaled = scaler.transform([features])

                # Predict the letter
                prediction = model.predict(features_scaled)[0]

                return jsonify({'letter': prediction, 'hand_detected': True})

        return jsonify({'letter': None, 'hand_detected': False})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)