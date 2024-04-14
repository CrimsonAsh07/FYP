import cv2
import os

def resize_to_average(images_path):
  """
  Resizes all images in a directory to the average dimensions of all images.

  Args:
      images_path: Path to the directory containing the images.
  """
  images = []
  total_width = 0
  total_height = 0
  prev_height = None
  prev_width = None
  mCount = 0
  checkUniform = False
  # Loop through all images in the directory
  for filename in os.listdir(images_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
      img = cv2.imread(os.path.join(images_path, filename))
      height, width, _ = img.shape
      if prev_height is None and prev_width is None:
        prev_width = width
        prev_height = height
      elif prev_width != width or prev_height != height:
        print("mismatch!", mCount)
        mCount+=1
      images.append((filename, img))
      total_width += width
      total_height += height

  # Check if any images were found
  if not images:
    print("No images found in the specified directory.")
    return

  # Calculate average dimensions
  average_width = int(total_width / len(images))
  average_height = int(total_height / len(images))
  average_width = 96
  average_height = 224
  # Resize and save all images

  if (mCount > 0 or not checkUniform):
    print("ye")
    for filename, img in images:
        resized_img = cv2.resize(img, (average_width, average_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(images_path, filename), resized_img)
        print(f"Resized and saved: {filename}")

# Replace 'path/to/your/images' with the actual path to your image directory

resize_to_average('D:/AmeenCLG/Sem 8/FYPCode/RE_ID VAE/data/bounding_box_test')