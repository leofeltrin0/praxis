import cv2
import os
import imageio

# Path to the directory containing the images
images_folder = 'output/blender_reder'

# Get a list of image filenames in the directory
image_filenames = os.listdir(images_folder)
image_filenames.sort()  # Sort filenames to ensure the correct order

# Set the frame size (you can change these values as needed)
frame_width = 1920
frame_height = 1080

# Create the video writer object
video_output_filename = 'output/blender_reder/output_video.mp4'
fps = 3.0  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_output_filename, fourcc, fps, (frame_width, frame_height))

# Iterate through the images and write each image as a frame into the video
IMG_NUM = 20
frames = []

for image_filename in image_filenames[:IMG_NUM]:
    image_path = os.path.join(images_folder, image_filename)
    image = cv2.imread(image_path)

    # Resize the image to fit the video frame size
    resized_image = cv2.resize(image, (frame_width, frame_height))

    # Write the image as a frame to the video
    video_writer.write(resized_image)
    frames.append(resized_image)

# Release the video writer
video_writer.release()

print("Video concatenation complete.")

with imageio.get_writer("output/blender_reder/output_video.gif", mode="I") as writer:
    for idx, frame in enumerate(frames):
        print("Adding frame to GIF file: ", idx + 1)
        writer.append_data(frame[...,[2,1,0]])