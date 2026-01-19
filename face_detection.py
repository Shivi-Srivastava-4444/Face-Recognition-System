import cv2

# Path to the Haar cascade XML file you downloaded
cascade_path = "haarcascade_frontalface_default.xml"

# Load the cascade
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError(f"Failed to load cascade from {cascade_path}")

# Open camera (0 is default webcam, try 1 if 0 doesn't work)
cam_index = 0
cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    raise IOError(f"Cannot open webcam (index {cam_index})")

cv2.namedWindow("FaceDetection", cv2.WINDOW_NORMAL)  # allow resizing

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to read frame from camera")
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detectMultiScale parameters:
    # scaleFactor = 1.3 (image size reduced by 30% at each image scale)
    # minNeighbors = 4 (higher => fewer false positives)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("FaceDetection", img)

    # Wait 10 ms for key press; if 'Esc' (27) pressed break
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
