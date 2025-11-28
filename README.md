---
hf_python_version: 3.10.0
title: GenSim
emoji: 📈
colorFrom: purple
colorTo: indigo
sdk: gradio
python_version: 3.10.0
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

## Preparations
1. Obtain an [OpenAI API Key](https://openai.com/blog/openai-api/)

## Usage 
0. Click Run-Example will simulate one example of pre-saved tasks in the task library and render videos.
1. Top-Down Model:
  0. Type in the desired task name in the box. Then GenSim will try to run through the pipeline to generate the task.
  1. The task name has the form word separated by a dash. **Example: 'place-blue-in-yellow' and 'align-rainbow-along-line'.**
2. Bottom-Up Model: No need to type in desired task. GenSim will try to generate novel tasks that are different from the task library.
3. Usage: Always click on "Setup/Reset Simulation" and then click "Run".

## Guideline
0. The first output is the current stage of the task generation pipeline.
1. The second output shows the generated code from Gen-Sim
2. If there are errors in the generation stage above, you will see an error log on the top right.
3. If the orange borders are still on, then the task is being simulated and rendered.
4. The rendered video will come out in a stream, i.e. it will render and re-render in a sequence. Each new update takes 15 seconds.


## Known Limitations
1. Code generation can fail or generate infeasible tasks. The success rate is around *0.5*. 
2. The low-level pick place primitive does not do collision checking and cannot pick up certain objects.
3. Top-down generation is typically more challenging if the task name is too vague or too distant from the primitives.


## Note
For GPT-4 model, each inference costs about *$\\$$0.03*. For GPT-3.5 model, each inference costs about *$\\$$0.005*. You can select which LLM model you would like to use.