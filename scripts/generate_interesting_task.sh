

for task in "rope-disentange" "move-piles-along-line" "rope-along-line" "rope-connect-cylinder" "rope-connect-corners"
    do
    	python gensim/run_simulation.py disp=False  prompt_folder=cliport_multistep_collaborative_prompt trials=20 \
    	 	save_memory=True load_memory=True task_description_candidate_num=10 use_template=True target_task_name=$task
	done
