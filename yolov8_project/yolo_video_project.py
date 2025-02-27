from ultralytics import YOLO
import cv2
import os

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with your custom helmet model if available

# Path to the video file
video_path = r"c:\Users\rames\Videos\Captures\he2.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)  # Open the video file

# Create a directory to save images if it doesn't exist
output_dir = r"C:\Users\rames\yolov8_project\images"
os.makedirs(output_dir, exist_ok=True)

# Assuming class_id 0 corresponds to helmets; adjust if your class IDs differ
helmet_class_id = 0  # Change this if your helmet class has a different ID

image_count = 0  # To keep track of captured images

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit if the video ends

    # Run inference on the current frame
    results = model(frame)

    # Initialize a flag to check for helmet detection
    helmet_detected = False
    img = frame.copy()  # Create a copy of the original frame for drawing

    # Iterate through detections
    for box in results[0].boxes:
        class_id = int(box.cls)  # Get the class ID of the detected object
        if class_id == helmet_class_id:  # Check if the detected class is a helmet
            helmet_detected = True  # Set flag if helmets are detected
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Unpack the bounding box

            # Adjust the bounding box to focus on the head area
            head_height_adjustment = int((y2 - y1) * 0.5)  # Adjust height to focus on the head
            head_y1 = y1  # Keep the top the same
            head_y2 = y1 + head_height_adjustment  # Reduce the height to focus on the head

            # Draw the bounding box around the head
            cv2.rectangle(img, (x1, head_y1), (x2, head_y2), (0, 255, 0), 2)  # Green box for helmets
            cv2.putText(img, "Helmet", (x1, head_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Add text based on detection
    if helmet_detected:
        cv2.putText(img, "Helmet Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green text
        # Save the frame if a helmet is detected
        image_name = f"helmet_detected_{image_count}.jpg"  # Create a unique name for the image
        cv2.imwrite(os.path.join(output_dir, image_name), frame)  # Save the frame as an image
        image_count += 1  # Increment the image count
    else:
        cv2.putText(img, "No Helmet Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red text

    # Resize the image for display if needed, maintaining aspect ratio
    display_height = 600  # Set your desired display height
    aspect_ratio = frame.shape[1] / frame.shape[0]
    display_width = int(display_height * aspect_ratio)
    img_resized = cv2.resize(img, (display_width, display_height))

    # Display the frame with detections
    cv2.imshow('Helmet Detection', img_resized)  # Show the resized frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
