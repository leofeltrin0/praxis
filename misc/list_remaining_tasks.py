import os

def check_missing_task_data(target_task):
	for existing_folder in os.listdir("data"):
		if target_task + "-train" == existing_folder:
			color_subdir = os.path.join('data', existing_folder, 'color')
			if os.path.exists(color_subdir) and  len(os.listdir(os.path.join('data', existing_folder, 'color'))) >= 40:
				return False
	return True

total_tasks = os.listdir("cliport/tasks") + os.listdir("cliport/generated_tasks")

total_tasks = [t.replace("_", "-")[:-3]  for t in total_tasks if 'pycache' not in t and 'init' not in t \
				and 'README' not in t and 'extended' not in t and 'gripper' not in t and 'primitive' not in t\
				and 'generated' not in t and 'camera' not in t and t != 'task']
print(total_tasks)
remaining_tasks = [t  for t in total_tasks if (check_missing_task_data(t)) ]
# print(f"run sh  scripts/generate_gpt_datasets.sh data {' '.join(remaining_tasks)}")

# print(f"sh scripts/traintest_scripts/train_test_multi_task_indistribution.sh data \
# 		'[{','.join(remaining_tasks)}]' gpt10_task_indomain"
# )
for t in total_tasks:
	print("sh scripts/train_test_single_task.sh data " + t)

s = ''
for t in total_tasks:
	s = s + f"python cliport/demos.py n=5 task={t} mode=test disp=False record.save_video=True +regenerate_data=True record.add_text=True +record.blender_render=True ;\n"
print(s)

s = ''
for t in total_tasks:
	s = s + f"cp -r data/{t}-test/videos output/code_video_website/{t}-videos\n"
print(s)


print("sh scripts/test_all_singletask.sh data \"" + ' '.join(total_tasks) +"\"")
# for t in ['color-specific-container-fill', 'build-two-circles', 'push-piles-into-letter', 'insert-blocks-lineup', 'align-pair-colored-blocks-along-line', 'color-blocks-in-cylinder-maze']:
# 	s = s + f"python cliport/demos.py n=5 task={t} mode=test disp=False record.save_video=True +regenerate_data=True record.add_text=True;"
# print(s)