# 🤟 Sign Language Translator

A real-time American Sign Language (ASL) translator that uses computer vision 
and machine learning to recognize hand gestures via webcam and convert them to text.

## 🌐 Live Demo
**[Try it here → https://huggingface.co/spaces/nic79cy/sign-language-translator](https://huggingface.co/spaces/nic79cy/sign-language-translator)**

---

## 📖 About The Project

Over 70 million deaf people worldwide use sign language as their primary form 
of communication. Yet most people don't understand sign language, creating a 
significant communication barrier in everyday life.

I built this tool to help bridge that gap — making it possible for anyone with 
a webcam to communicate using ASL, without needing a human interpreter.

---

## ✨ Features

- 🎥 Real-time hand detection via webcam
- 🧠 Neural network trained on 5,200 samples
- 🔤 Recognizes full ASL alphabet (A–Z)
- 📝 Builds words letter by letter automatically
- 🔊 Text-to-speech output
- 🌐 Runs entirely in the browser — no installation needed

---

## 🛠️ How It Works

**Webcam → MediaPipe (21 hand landmarks) → Neural Network → Predicted Letter**

1. **Hand Detection** — MediaPipe by Google detects 21 key points on the hand in real time
2. **Normalization** — Landmark positions are normalized relative to the wrist, making predictions consistent regardless of hand position or distance
3. **Prediction** — A 3-layer neural network (256→128→64 neurons) classifies the gesture into one of 26 ASL letters
4. **Smoothing** — A 15-frame stability buffer ensures only confident, stable predictions are displayed

---

## 🧠 Machine Learning Details

| Property | Value |
|---|---|
| Model | MLPClassifier (Neural Network) |
| Architecture | 256 → 128 → 64 neurons |
| Training samples | 5,200 (200 per letter) |
| Features | 42 (x,y coordinates of 21 landmarks) |
| Test accuracy | 100% (on own hand — generalizes to similar users) |
| Preprocessing | StandardScaler normalization |

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Hand tracking | MediaPipe |
| Machine learning | scikit-learn |
| Image processing | OpenCV |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Docker, Hugging Face Spaces |

---

## 🚧 Challenges I Faced

**1. Python version compatibility**
MediaPipe didn't support Python 3.14, requiring migration from the old 
mp. solutions API to the new Tasks API, which had almost no documentation 
for beginners.

**2. Similar letter confusion**
Letters like M/N and R/U are visually similar in landmark space. Switching 
from Random Forest to a Neural Network and adding StandardScaler preprocessing 
significantly improved accuracy on these edge cases.

**3. Deployment system libraries**
MediaPipe requires OpenGL libraries (libEGL, libGL) that aren't installed 
on cloud servers by default. Debugging this required reading deep into 
MediaPipe's C++ bindings and Docker configuration.

**4. Binary file in git history**
The trained model (model.pkl) was rejected by Hugging Face as a binary file. 
The solution was to remove it from git history and make the app auto-train 
on first startup using the dataset.

---

## 📈 Future Improvements

- [ ] Support full words and phrases, not just letters
- [ ] Collect data from multiple people for better generalization
- [ ] Add support for other sign languages (BSL, LSF)
- [ ] Deploy a mobile version using TensorFlow.js
- [ ] Partner with schools for the deaf for real-world testing

---

## 🚀 Run Locally

    git clone https://github.com/Nictheprogrammer/sign-language-translator
    pip install flask opencv-python mediapipe scikit-learn numpy
    python app.py

Then open your browser at http://localhost:5000

---

## 👨‍💻 About

Built by **Nicolas** — a high school 
student from Cyprus who is passionate about using AI to solve real-world 
accessibility problems.

