import cv2
image_path = "test_data/enhance4.jpg"  # Replace with your actual image path
from Zero_DCE import lowlight_test_frame

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')

# Read the image
img = cv2.imread(image_path)

# Check if image is read successfully
if img is None:
    print("Error: Could not read image from", image_path)
else:
    # Process the image
    processed_image =lowlight_test_frame.lowlight(img)
    processed_image =  cv2.GaussianBlur(processed_image, (7, 7), 0) 
    # Get the original filename with extension (e.g., "image.jpg")
    filename = image_path.split('/')[-1]

    # Construct the new filename with "process_" prefix (e.g., "process_image.jpg")
    new_filename = f"process_{filename}"

    # Save the processed image
    result = model(processed_image, classes=[0])
    result2 = model(img, classes=[0])
    annotated_frame = result[0].plot()
    annotated_frame2 = result2[0].plot()
    cv2.imshow("",annotated_frame )
    cv2.waitKey(0)

    cv2.imwrite("test_data/"+new_filename, processed_image)
    cv2.imwrite("test_data/annotated"+new_filename, annotated_frame)
    cv2.imwrite("test_data/annotated"+filename, annotated_frame2)
    print(f"Image saved as: {new_filename}")

# Release resources
cv2.destroyAllWindows()