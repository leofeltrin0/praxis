import cv2
import numpy as np
import IPython
import os

output_folder = "output/output_gifs/"
output_img_folder = "output/output_imgs/"

total_tasks = [s  for s in sorted(os.listdir(output_folder))  if s.endswith("mp4") and not s.startswith("grid")]
print(total_tasks)

# Load videos
videos = [cv2.VideoCapture(os.path.join(output_folder, s))
      for s in total_tasks ]

# Read all frames
video_frames = [[] for _ in range(len(videos))]
for i, video in enumerate(videos):
      frame = None

      while True:
            prev_frame = frame
            ret, frame = video.read()
            if not ret:
                  break

      # last frame
      try:
            frame = prev_frame[:550,:]
            print(f"write {output_img_folder}/{total_tasks[i]}_img.png")
            cv2.imwrite(f"{output_img_folder}/{total_tasks[i]}_img.png", frame)
      except:
            print("failed:", total_tasks[i])

