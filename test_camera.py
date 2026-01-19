import cv2

print("Testing OpenCV setup...")

# Check OpenCV version
print("OpenCV version:", cv2.__version__)

# Test cascade loading
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("❌ Error: Could not load Haar Cascade file!")
else:
    print("✅ Haar Cascade loaded successfully.")

# Test camera access
for i in range(3):
    cam = cv2.VideoCapture(i)
    if cam.isOpened():
        print(f"✅ Camera index {i} opened successfully.")
        ret, frame = cam.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            print("Press any key to close this window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cam.release()
        break
    else:
        print(f"❌ Camera index {i} not available.")
