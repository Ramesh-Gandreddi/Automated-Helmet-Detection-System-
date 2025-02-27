from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8 model
model = YOLO('best.pt')  # Load the model

# Run inference on an image
image_path = r'C:\Users\rames\OneDrive\Desktop\Ai project\dataset\images\BikesHelmets49.png'  # Replace with your image path
results = model(image_path)

# Iterate through results
for result in results:
    if result.boxes:
        img = result.plot()  # Get annotated image with bounding boxes
    else:
        img = cv2.imread(image_path)  # Load original image if no helmets detected

        # Add smaller text for "No Helmet Detected"
        cv2.putText(
            img,
            "No Helmet Detected",  # Text to display
            (10, 30),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            0.7,  # Slightly smaller font scale
            (0, 0, 255),  # Red color
            2,  # Thickness of 2 for better visibility
            cv2.LINE_AA  # Anti-aliased for smooth text

        )

    # Get the original dimensions of the image
    original_height, original_width = img.shape[:2]

    # Set a large display window size
    max_display_width, max_display_height = 900, 800

    # Calculate scaling factor to fit the image within the display window
    scale = min(max_display_width / original_width, max_display_height / original_height)
    new_width, new_height = int(original_width * scale), int(original_height * scale)

    # Resize the image to fit within the window while maintaining aspect ratio
    resized_img = cv2.resize(img, (new_width, new_height))

    # Create a named window and allow it to resize
    cv2.namedWindow('Helmet Detection', cv2.WINDOW_NORMAL)

    # Set the window size to match the resized image dimensions
    cv2.resizeWindow('Helmet Detection', new_width, new_height)

    # Display the image in the window
    cv2.imshow('Helmet Detection', resized_img)

    # Wait for a key press to close the window
    cv2.waitKey(0)

# Destroy all windows after display
cv2.destroyAllWindows()


