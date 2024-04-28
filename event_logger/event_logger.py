import os
import datetime

def create_unique_folder_and_log(base_dir="logs"):
  """
  Creates a unique folder inside the specified base directory and a log file within it.

  Args:
      base_dir: The base directory path where unique folders will be created. Defaults to "logs".

  Returns:
      A tuple containing the path to the created folder and the path to the log file.
  """

  # Get current date and time for folder name
  current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  # Create a unique folder name using timestamp
  unique_folder_name = f"{current_time}"

  # Create the base directory if it doesn't exist
  os.makedirs(base_dir, exist_ok=True)  # Create directories recursively if needed

  # Combine base directory and unique folder name
  folder_path = os.path.join(base_dir, unique_folder_name)

  # Create the unique folder
  os.makedirs(folder_path)

  # Create the log file path within the folder
  log_file_path = os.path.join(folder_path, "surveillance_log.txt")

  return folder_path, log_file_path

# Example usage


def write_log(log_file_path, message):
  formatted_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  with open(log_file_path, "a+") as log_file:
    log_file.write(f"{formatted_timestamp} - {message}\n")