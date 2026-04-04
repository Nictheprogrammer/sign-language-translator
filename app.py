from flask import Flask, render_template, request, jsonify
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import base64
import numpy as np
import cv2
import os
import csv
import random
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def train_model():
    print("Training model from dataset...")
    rows = []
    with open('dataset.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append(row)

    random.shuffle(rows)
    X, y = [], []
    for row in rows:
        X.append([float(v) for v in row[:-1]])
        y.append(row[-1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_scaled, y)

    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)

    print("Model trained and saved!")
    return model, scaler

# Load or train model
if os.path.exists('model.pkl'):
    print("Loading model...")
    with open('model.pkl', 'rb') as f:
        saved = pickle.load(f)
        model = saved['model']
        scaler = saved['scaler']
    print("Model loaded!")
else:
    model, scaler = train_model()

# Load hand detection model
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand model...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Download done!")

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
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                features = normalize(hand_landmarks)
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)[0]
                confidence = model.predict_proba(features_scaled).max()

                # Only return prediction if confidence is above 70%
                if confidence < 0.7:
                    return jsonify({'letter': None, 'hand_detected': False})

                return jsonify({
                    'letter': prediction,
                    'hand_detected': True,
                    'confidence': round(float(confidence) * 100, 1)
                })

        return jsonify({'letter': None, 'hand_detected': False})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)