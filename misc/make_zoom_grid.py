import cv2
import numpy as np
import IPython
import os

# Define the grid dimensions
num_rows = 6

output_folder = "output/output_gifs/"
total_tasks = os.listdir(output_folder)
# Load videos
videos = [cv2.VideoCapture(os.path.join(output_folder, s))
      for s in total_tasks if s.endswith("mp4") and not s.startswith("grid")]
num_cols = len(videos) // num_rows + 1

print(f"num_rows: {num_rows} num_cols: {num_cols}")

# Get the dimensions of the videos
video_width = 640 # int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = 480 # int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the output frame
output_width = video_width * num_cols
output_height = video_height * num_rows

output_filename = output_folder + 'gslide_output2.mp4'
zoomed_output_filename = output_folder + 'zoom_gslide_output.mp4'
output_video  = cv2.VideoCapture(output_filename)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
resolution_factor = 2
grid_video = cv2.VideoWriter(zoomed_output_filename, fourcc, 30.0, (resolution_factor * video_width, resolution_factor * video_height))

print(video_width, video_height)
stop = 100
max_length = 500
# Create the zoom-out effect
for idx in range(max_length):
    if idx < stop:
        ratio = 0.23
    elif idx > max_length - stop:
        ratio = 1
    else:
        ratio = 0.23 + 0.8 * float(idx - stop) / (max_length - 2 * stop)
    print(idx)
    ret, frame = output_video.read()
    if not ret:
        break

    # Apply the zoom-out effect by resizing the frame with the current ratio
    center = frame.shape[0] // 2, frame.shape[1] // 2
    size = int(ratio * center[0]), int(ratio * center[1])
    zoomed_frame = frame[center[0]-size[0]:center[0]+size[0],center[1]-size[1]:center[1]+size[1]]

    # And then resize to video image size
    resized_image = cv2.resize(zoomed_frame, (resolution_factor * video_width, resolution_factor * video_height))

    # Display the zoomed frame
    cv2.imshow('Zoom Out Grid', resized_image)
    grid_video.write(resized_image)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the grid video and close all windows
grid_video.release()
output_video.release()
cv2.destroyAllWindows()