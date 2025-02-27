# Import necessary libraries
from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model (replace 'path_to_your_model.pt' with your model path)
model = YOLO(r"C:\Users\rames\Downloads\weights\weights\best.pt")

# Initialize the video capture from the webcam (0 for default webcam)
cap = cv2.VideoCapture(r"C:\Users\rames\OneDrive\Desktop\yolov8_project\he2.mp4")

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to continuously get frames from the video feed
while True:
    # Capture each frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 detection on the frame
    results = model(frame)

    # Render the detected boxes and labels on the frame
    annotated_frame = results[0].plot()  # YOLO's built-in plotting for annotations

    # Display the annotated frame
    cv2.imshow('Helmet Detection', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
