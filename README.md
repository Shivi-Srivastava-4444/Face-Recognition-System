A real-time face recognition and identification system built with Python, OpenCV, and dlib. This project detects faces from a live webcam feed or video file and matches them against a pre-encoded database of known individuals using Deep Learning feature extraction.

🚀 Features
Real-time Detection: High-speed face detection using Haar Cascades or HOG-based models.
Deep Learning Encoding: Converts facial features into a 128-dimension vector to ensure high accuracy even with different lighting or angles.
Multiple Face Tracking: Ability to identify and track multiple individuals simultaneously.
Database Management: Simple script to add new faces by just dropping an image into a folder.

🛠️ Technical Stack
Language: Python 3.x
Computer Vision: OpenCV (cv2)
Face Processing: dlib, face_recognition library
Data Handling: NumPy, Pandas

⚙️ How It Works
The system follows a three-step pipeline:
Face Detection: Locates the face in the frame.
Feature Extraction: The dlib model extracts unique facial landmarks (eyes, nose, jawline).
Matching: The system calculates the Euclidean Distance between the live face and the stored encodings. If the distance is below a specific threshold (e.g., 0.6), a match is confirmed.
