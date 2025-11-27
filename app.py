
from hydra.core.global_hydra import GlobalHydra
import gradio as gr
import os
import hydra
import random

import re
import openai
import IPython
import time
import pybullet as p
import traceback
from datetime import datetime
from pprint import pprint
import cv2
import re
import random
import json

from gensim.agent import Agent
from gensim.critic import Critic
from gensim.sim_runner import SimulationRunner
from gensim.memory import Memory
from gensim.utils import set_gpt_model, clear_messages


class DemoRunner:

    def __init__(self):
        self._env = None
        GlobalHydra.instance().clear()
        hydra.initialize(version_base="1.2", config_path='cliport/cfg')
        self._cfg = hydra.compose(config_name="data")

    def setup(self, api_key, model_choice):
        cfg = self._cfg
        cfg['gpt_model'] = model_choice
        openai.api_key = api_key
        cfg['model_output_dir'] = 'temp'
        cfg['prompt_folder'] = 'bottomup_task_generation_prompt'
        set_gpt_model(cfg['gpt_model'])
        cfg['load_memory'] = False
        cfg['use_template'] = True
        cfg['task_description_candidate_num'] = 2
        cfg['record']['save_video'] = True

        print("cfg = ", cfg)
        memory = Memory(cfg)
        agent = Agent(cfg, memory)
        critic = Critic(cfg, memory)
        self.simulation_runner = SimulationRunner(cfg, agent, critic, memory)

        info = '### Configuração concluída '

        return info

    def setup_top_down(self, api_key, target_task_name, model_choice):
        cfg = self._cfg
        cfg['gpt_model'] = model_choice
        openai.api_key = api_key
        cfg['model_output_dir'] = 'temp'
        cfg['prompt_folder'] = 'topdown_task_generation_prompt'
        set_gpt_model(cfg['gpt_model'])
        cfg['load_memory'] = True
        cfg['use_template'] = True
        cfg['target_task_name'] = target_task_name
        cfg['task_description_candidate_num'] = 10
        cfg['record']['save_video'] = True

        print("cfg = ", cfg)
        memory = Memory(cfg)
        agent = Agent(cfg, memory)
        critic = Critic(cfg, memory)
        self.simulation_runner = SimulationRunner(cfg, agent, critic, memory)

        info = '### Configuração concluída '

        return info

    def run(self, instruction, progress):
        cfg = self._cfg
        cfg['target_task_name'] = instruction

        # self._env.cache_video = []
        self.simulation_runner._md_logger = ''
        # progress(0.2)
        yield "Gerando tarefa ==>", "",None, None
        yield from self.simulation_runner.task_creation()
        yield from self.simulation_runner.simulate_task()

    def run_example(self):
        cfg = self._cfg

        # self._env.cache_video = []
        self.simulation_runner._md_logger = ''
        # progress(0.2)
        yield "Gerando tarefa ==>", "", None, None

        t1 = time.time()
        yield from self.simulation_runner.example_task_creation()
        yield from self.simulation_runner.simulate_task()
        # self.simulation_runner.example_task_creation()
        # self.simulation_runner.simulate_task()

        t2 = time.time()
        print("run example cost = ", t2 - t1, " s")


def setup(api_key, option_choice, model_choice, target_task_name):
    print(option_choice)
    if not api_key:
        return 'Por favor, insira sua chave da API OpenAI!', None

    if model_choice is None:
        return 'Escolha um modelo!', None
    if option_choice is None:
        return 'Escolha um modo!', None
    demo_runner = DemoRunner()

    if option_choice == 'top-down':
        info = demo_runner.setup_top_down(api_key, target_task_name, model_choice) + option_choice
    # elif option_choice == 'bottom-up':
    #     info = demo_runner.setup(api_key, model_choice) + option_choice
    else:
        raise NotImplementedError
    return info, demo_runner



def run(instruction, demo_runner, progress=gr.Progress()):
    yield from demo_runner.run(instruction, progress=progress)

def run_example():
    demo_runner = DemoRunner()
    demo_runner.setup(1, "gpt-4")
    yield from demo_runner.run_example()


if __name__ == '__main__':
    os.environ['GENSIM_ROOT'] = os.getcwd()

    with gr.Blocks() as demo:
        state = gr.State(None)

        gr.Markdown('# Demo Interativo')
        with gr.Row():
            with gr.Column():


                btn_example_run = gr.Button("Executar exemplo (não precisa de chave OpenAI)")
                with gr.Row():
                    inp_api_key = gr.Textbox(label='OpenAI API Key (não armazenamos este valor)', lines=1)

                model_choice = gr.Radio(["gpt-3.5-turbo-16k", "gpt-4"], label="Qual modelo?", interactive=True)
                option_choice = gr.Radio(["top-down"], label="Qual modo?", interactive=True)
                inp_instruction = gr.Textbox(label='Nome da tarefa alvo (se usar top-down)', lines=1)
                info_setup = gr.Markdown(label='Status da configuração')
                btn_setup = gr.Button("Configurar/Resetar simulação")
                btn_run = gr.Button("Executar (pode levar 30+ segundos)")
            # with gr.Column():

        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                progress = gr.Markdown(label='Progresso')
                generated_task = gr.Markdown(label='Tarefa gerada')
                log = gr.HTML(label='Log')
                generated_asset = gr.Markdown(label='Asset gerado')
                generated_code = gr.Code(label='Código gerado',  language="python", interactive=True)
                video_run = gr.Video(label='Vídeo da simulação', autoplay=True)
        btn_setup.click(
            setup,
            inputs=[inp_api_key, option_choice, model_choice, inp_instruction],
            outputs=[info_setup, state]
        )
        btn_run.click(
            run,
            inputs=[inp_instruction, state],
            outputs=[progress, log, generated_code, video_run]
        )

        btn_example_run.click(
            run_example,
            inputs=[],
            outputs=[progress, log, generated_code, video_run]
        )


    demo.queue().launch(show_error=True)
