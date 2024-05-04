import cv2
image_path = "test_data/182.jpg"  # Replace with your actual image path
from Zero_DCE import lowlight_test_frame
# Read the image
img = cv2.imread(image_path)

# Check if image is read successfully
if img is None:
    print("Error: Could not read image from", image_path)
else:
    # Process the image
    processed_image =lowlight_test_frame.lowlight(img)

    # Get the original filename with extension (e.g., "image.jpg")
    filename = image_path.split('/')[-1]

    # Construct the new filename with "process_" prefix (e.g., "process_image.jpg")
    new_filename = f"process_{filename}"

    # Save the processed image
    cv2.imwrite("test_data/"+new_filename, processed_image)
    print(f"Image saved as: {new_filename}")

# Release resources
cv2.destroyAllWindows()