import threading
import cv2
from deepface import DeepFace

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Load the reference image
reference_img = cv2.imread("reference.jpg")
if reference_img is None:
    print("Error: Could not load reference image")
    exit()

def check_face(frame): 
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False 

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if face_match:
            cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Match!", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()