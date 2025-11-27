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
video_width = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the output frame
output_width = video_width * num_cols
output_height = video_height * num_rows

output_filename = output_folder + 'grid_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_filename, fourcc, 30.0, (output_width, output_height))


max_length = 200

# Read all frames
video_frames = [[] for _ in range(len(videos))]
for i, video in enumerate((videos)):
    while True:
        ret, frame = video.read()
        if not ret:
            break
        video_frames[i].append(frame)
    if len(video_frames) == 0 :
        continue
    # print(max_length, len(video_frames[i]))
    repeat_ratio = max_length // len(video_frames[i])
    left_ratio = max_length % len(video_frames[i])

    video_frames[i] = video_frames[i] * repeat_ratio
    video_frames[i] += video_frames[i][:left_ratio]
# Pad with repeated video

video_frames = [v for v in video_frames if len(v) == max_length]

# Resize and arrange the frames
print(len(video_frames), len(video_frames[0]))

for j, video_frame in enumerate(zip(*video_frames)):
    output_frame = 255 * np.ones((output_height, output_width, 3), np.uint8)
    for i, frame in enumerate(video_frame):
        # Resize the frame to a smaller size for the zoom-out effect

        # Calculate the row and column indices for placing the frame in the output frame
        row = i // num_cols
        col = i % num_cols

        # Calculate the coordinates for placing the resized frame in the output frame
        x = col * (video_width  )
        y = row * (video_height )

        # Place the resized frame in the output frame
        output_frame[y:y+frame.shape[0], x:x+frame.shape[1]] = frame
    output_video.write(output_frame)

output_video.release()
zoomed_output_filename = output_folder + 'grid_video_zoomed.mp4'
output_video  = cv2.VideoCapture(output_filename)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
grid_video = cv2.VideoWriter(zoomed_output_filename, fourcc, 30.0, (video_width, video_height))


stop = 50

# Create the zoom-out effect
for idx in range(max_length):
    if idx < stop:
        ratio = 0.2
    else:
        ratio = 0.2 + 0.8 * float(idx - stop) / (max_length - stop)

    ret, frame = output_video.read()
    if not ret:
        break

    # Apply the zoom-out effect by resizing the frame with the current ratio
    center = frame.shape[0] // 2, frame.shape[1] // 2
    size = int(ratio * center[0]), int(ratio * center[1])
    zoomed_frame = frame[center[0]-size[0]:center[0]+size[0],center[1]-size[1]:center[1]+size[1]]
     # cv2.resize(frame, None, fx=ratio, fy=ratio)

    # And then resize to video image size
    resized_image = cv2.resize(zoomed_frame, (video_width, video_height))

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