import os
import shutil
def copy_and_rename_images(source_folder, destination_folder):
  """
  Copies and renames images in a folder based on image number.

  Args:
    source_folder: Path to the folder containing the original images.
    destination_folder: Path to the folder where renamed copies will be placed.
  """


  # Get all image files in the folder
  image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

  # Sort image files by name (assuming numeric sort)
  image_files.sort()

  current_image_number = None
  image_group = []
  n_images = 0
  for image_file in image_files:
    # Extract image number from first 4 digits
    image_number = int(image_file[:4])

    if image_number != current_image_number:
      # New image number, process previous group if any
      if current_image_number is not None:
        process_image_group(source_folder, destination_folder, image_group,current_image_number)
      current_image_number = image_number
      image_group = []
    image_group.append(image_file)

  # Process the last group
  process_image_group(source_folder, destination_folder, image_group,current_image_number)
def process_image_group(source_folder, destination_folder, image_group, current_image_number):
  """
  Processes a group of images with the same image number.

  Args:
    source_folder: Path to the folder containing the original images.
    destination_folder: Path to the folder where renamed copies will be placed.
    image_group: List of image filenames in the group.
  """
  n = len(image_group)

  # Check if even number of images
#   if n % 2 != 0:
#     print(f"Warning: Uneven number of images for number {current_image_number}")
#     return

 
  if(n%2):
    ln = n-3
  else:
    ln = n-2
  for j in range(0,ln,2):
    # Construct new filenames
    new_filename1 = f"{current_image_number}_{j + 1}_1.jpg"
    new_filename2 = f"{current_image_number}_{j + 1}_2.jpg"

    # Copy and rename images
    source_path1 = os.path.join(source_folder, image_group[j])
    dest_path1 = os.path.join(destination_folder, new_filename1)
    shutil.copy2(source_path1, dest_path1)  # Use copy2 to preserve metadata

    source_path2 = os.path.join(source_folder, image_group[j+1])
    dest_path2 = os.path.join(destination_folder, new_filename2)
    shutil.copy2(source_path2, dest_path2)

# Example usage: Assuming source folder is "C:/images" and destination is "C:/renamed_images"


source_path ="D:/AmeenCLG/Sem 8/FYPCode/RE_ID VAE/data-test/bounding_box_test"
dest_path = "D:/AmeenCLG/Sem 8/FYPCode/RE_ID VAE/data-pair/bounding_box_test2"
copy_and_rename_images(source_path,dest_path)