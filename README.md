# sign-language-translator

## Overview
A real-time sign language translator that uses a webcam to detect hand gestures and convert them into text and speech. Built using computer vision and machine learning.

## Features
- Recognizes 36 static signs (A-Z and 0-9) using Random Forest
- Recognizes dynamic gestures (HELLO, THANKS, BYE etc.) using LSTM neural network
- Converts detected signs into spoken words using text-to-speech
- Real-time detection with confidence scoring

## Tools & Technologies
- **Python** — Core programming language
- **MediaPipe** — Hand landmark detection (21 points per hand)
- **Random Forest** — Static gesture classification
- **LSTM Neural Network (Keras)** — Dynamic gesture recognition
- **OpenCV** — Real-time webcam processing
- **pyttsx3** — Text-to-speech conversion

## How It Works
1. Webcam captures hand in real time
2. MediaPipe detects 21 hand landmarks
3. Features are extracted and normalized
4. Static model classifies letters/digits
5. LSTM model classifies motion-based gestures
6. Detected sign is displayed and spoken aloud

## Dataset
- ASL dataset for static signs
- Custom collected dataset for dynamic gestures



