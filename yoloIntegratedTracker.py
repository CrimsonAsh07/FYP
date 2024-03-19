import os
import cv2
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Directory containing images
images_dir = "./enhanced_frames"


processed_files = []


def main():
    ideal_wait_time = 1000 / 10  # milliseconds (1 second / 24 FPS)

    while True:
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        new_files = [f for f in image_files if f not in processed_files]

        for image_file in new_files:
            start_time = time.time()  # Record start time before processing

            # Read the image
            frame = cv2.imread(os.path.join(images_dir, image_file))

            # Run YOLOv8 tracking on the frame
            results = model.track(frame, persist=True)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # milliseconds

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Use dynamic wait time based on processing
            wait_time = max(0, ideal_wait_time - processing_time)
            if cv2.waitKey(int(ideal_wait_time)) & 0xFF == ord("q"):
                break

            # Add processed file to the list
            processed_files.append(image_file)

        # Check for new images every second (not strictly necessary now)
        # time.sleep(1)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    # Close the display window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()










# import os
# import cv2
# import time
# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')

# # Directory containing images
# images_dir = "./captured_frames"


# processed_files = []



# def main():
#     while True:
#         image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        
#         new_files = [f for f in image_files if f not in processed_files]

#         for image_file in new_files:
#             # Read the image
#             frame = cv2.imread(os.path.join(images_dir, image_file))

#             # Run YOLOv8 tracking on the frame
#             results = model.track(frame, persist=True)

#             # Visualize the results on the frame
#             annotated_frame = results[0].plot()

#             # Display the annotated frame
#             cv2.imshow("YOLOv8 Tracking", annotated_frame)

#             # Break the loop if 'q' is pressed
#             # if cv2.waitKey(1) & 0xFF == ord("q"):
#             #     break

#             # Add processed file to the list
#             processed_files.append(image_file)

#         # Check for new images every second

#         # if cv2.waitKey(0) & 0xFF == ord("q"):
#         #             break

#     # Close the display window
#         cv2.destroyAllWindows()




# if __name__ == '__main__':
#     main()



# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO('yolov8n.pt')

# # Open the video file
# video_path = "path/to/video.mp4"
# cap = cv2.VideoCapture(video_path)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         results = model.track(frame, persist=True)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Tracking", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()