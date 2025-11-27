import os
import json

# generated_tasks = json.load(open('prompts/data/generated_tasks.json'))
generated_tasks = json.load(open('prompts/data/base_tasks.json'))

generated_tasks = ([g['task-name'] for k, g in generated_tasks.items()])

# generated_tasks = [ 'build-house' ]
output_directory = 'output/output_gifs'
for data_folder in os.listdir('data'):
    folder = os.path.join('data', data_folder, 'videos')
    # print(data_folder)
    if data_folder[:-5]  in generated_tasks:
        if os.path.exists(folder):
            try:
                input_file = os.path.join(folder, "input.txt")
                with open(input_file, "w") as file:
                    video_files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
                    for video_file in video_files:
                        file.write(f"file '{ video_file}'\n")

                # Concatenate the videos within the subfolder
                concatenated_file = os.path.join(output_directory, f"{data_folder}.mp4")
                ffmpeg_concat_command = f"ffmpeg -f concat -safe 0 -i {input_file} -c copy {concatenated_file}"
                os.system(f"rm {concatenated_file}")
                os.system(ffmpeg_concat_command)

                # Convert the concatenated video to a GIF
                gif_file = os.path.join(output_directory, f"{data_folder}.gif")
                os.system(f"rm {gif_file}")
                ffmpeg_gif_command = f"ffmpeg -i {concatenated_file} -frames:v 800 {gif_file}"
                os.system(ffmpeg_gif_command)
                # os.system(f"rm {concatenated_file}")

            except Exception as e:
                print(e)