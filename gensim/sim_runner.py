import numpy as np
import os
import IPython
from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment

import imageio

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter, TerminalFormatter
import gradio
import time
import random
import json
import traceback
import re
from gensim.utils import (
    mkdir_if_missing,
    save_text,
    save_stat,
    compute_diversity_score_from_assets,
    add_to_txt
)
import pybullet as p
from gensim.code_fixer import attempt_code_repair

POSE_PATTERN = re.compile(
    r"\(\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)\s*,\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)\s*\)"
)

class SimulationRunner:
    """ the main class that runs simulation loop """
    def __init__(self, cfg, agent, critic, memory):
        self.cfg = cfg
        self.agent = agent
        self.critic = critic
        self.memory = memory
        self.log = ""

        # statistics
        self.syntax_pass_rate = 0
        self.runtime_pass_rate = 0
        self.env_pass_rate = 0
        self.curr_trials = 0

        self.prompt_folder = f"prompts/{cfg['prompt_folder']}"
        self.chat_log = memory.chat_log
        self.task_asset_logs = []

        # All the generated tasks in this run.
        # Different from the ones in online buffer that can load from offline.
        self.generated_task_assets = []
        self.generated_task_programs = []
        self.generated_task_names = []
        self.generated_tasks = []
        self.passed_tasks = [] # accepted ones
        self.auto_fix_attempts = cfg.get('auto_fix_attempts', 0)
        pose_validation_cfg = cfg.get('pose_validation', {})
        self.pose_validation_enabled = pose_validation_cfg.get('enabled', False)
        self.pose_validation_max_retries = pose_validation_cfg.get('max_retries', 0)
        xy_bounds = pose_validation_cfg.get('xy_bounds', [0.0, 1.0])
        z_bounds = pose_validation_cfg.get('z_bounds', [0.0, 1.0])
        self.pose_xy_bounds = (xy_bounds[0], xy_bounds[1])
        self.pose_z_bounds = (z_bounds[0], z_bounds[1])

    def print_current_stats(self):
        """ print the current statistics of the simulation design """
        print("=========================================================")
        print(f"{self.cfg['prompt_folder']} Trial {self.curr_trials} SYNTAX_PASS_RATE: {(self.syntax_pass_rate / (self.curr_trials)) * 100:.1f}% RUNTIME_PASS_RATE: {(self.runtime_pass_rate / (self.curr_trials)) * 100:.1f}% ENV_PASS_RATE: {(self.env_pass_rate / (self.curr_trials)) * 100:.1f}%")
        print("=========================================================")

    def save_stats(self):
        """ save the final simulation statistics """
        self.diversity_score = compute_diversity_score_from_assets(self.task_asset_logs, self.curr_trials)
        save_stat(self.cfg, self.cfg['model_output_dir'], self.generated_tasks, self.syntax_pass_rate / (self.curr_trials),
                self.runtime_pass_rate / (self.data_pathcurr_trials), self.env_pass_rate / (self.curr_trials), self.diversity_score)
        print("Model Folder: ", self.cfg['model_output_dir'])
        print(f"Total {len(self.generated_tasks)} New Tasks:", [task['task-name'] for task in self.generated_tasks])
        try:
            print(f"Added {len(self.passed_tasks)}  Tasks:", self.passed_tasks)
        except:
            pass

    def _format_html_traceback(self, trace_str):
        return highlight(f"{str(trace_str)}", PythonLexer(), HtmlFormatter())

    def _can_attempt_fix(self, attempt_idx):
        return attempt_idx < self.auto_fix_attempts

    def _reset_physics(self):
        if p.isConnected():
            p.disconnect()

    def _apply_auto_fix(self, error_trace, attempt_idx):
        fix = attempt_code_repair(
            cfg=self.cfg,
            task_metadata=self.generated_task,
            current_code=self.generated_code,
            error_log=error_trace,
            attempt_idx=attempt_idx,
            interaction_log=self.chat_log,
        )

        if not fix:
            return False

        new_code, new_task_name = fix
        if not new_code:
            return False
        is_valid, reason = self._validate_generated_code(new_code)
        if not is_valid:
            print("Auto-fix rejected due to invalid pose:", reason)
            self.log = self._format_html_traceback(reason)
            return False

        self.generated_code = new_code
        if new_task_name:
            self.curr_task_name = new_task_name

        if self.generated_task_programs:
            self.generated_task_programs[-1] = new_code

        save_text(
            self.cfg['model_output_dir'],
            f"{self.generated_task_name}_autofix_attempt_{attempt_idx + 1}",
            new_code,
        )
        return True

    def _validate_generated_code(self, code: str):
        if not self.pose_validation_enabled:
            return True, ""

        matches = POSE_PATTERN.findall(code)
        if not matches:
            return True, ""

        invalid_entries = []
        for pose in matches:
            try:
                x, y, z = map(float, pose[:3])
            except ValueError:
                continue
            if not (self.pose_xy_bounds[0] <= x <= self.pose_xy_bounds[1]):
                invalid_entries.append(f"x={x:.3f} fora dos limites {self.pose_xy_bounds}")
            if not (self.pose_xy_bounds[0] <= y <= self.pose_xy_bounds[1]):
                invalid_entries.append(f"y={y:.3f} fora dos limites {self.pose_xy_bounds}")
            if not (self.pose_z_bounds[0] <= z <= self.pose_z_bounds[1]):
                invalid_entries.append(f"z={z:.3f} fora dos limites {self.pose_z_bounds}")

            if len(invalid_entries) >= 3:
                break

        if invalid_entries:
            reason = "Validação de pose falhou: " + "; ".join(invalid_entries)
            return False, reason

        return True, ""

    def example_task_creation(self):
        """ create the task through interactions of agent and critic """
        self.task_creation_pass = True
        mkdir_if_missing(self.cfg['model_output_dir'])

        try:
            start_time = time.time()

            self.generated_task = {'task-name': 'TASK_NAME_TEMPLATE', 'task-description': 'TASK_STRING_TEMPLATE', 'assets-used': ['ASSET_1', 'ASSET_2', Ellipsis]}
            print("generated_task\n", self.generated_task)
            yield "Tarefa gerada ==>", "", None, None
            self.generated_asset = self.agent.propose_assets()
            # self.generated_asset = {}
            print("generated_asset\n", self.generated_asset)
            yield "Tarefa gerada ==> Asset gerado ==> ", "", None, None
            yield "Tarefa gerada ==> Asset gerado ==> API revisada ==> ", "", None, None
            yield "Tarefa gerada ==> Asset gerado ==> API revisada ==> Erros revisados ==> ", "", None, None

            online_code_buffer = {}
            for task_file in json.load(open(os.path.join('prompts/data', "generated_task_codes.json"))):
                if os.path.exists("cliport/generated_tasks/" + task_file):
                    online_code_buffer[task_file] = open("cliport/generated_tasks/" + task_file).read()

            random_task_file = random.sample(list(online_code_buffer.keys()), 1)[0]
            class_def = [line for line in online_code_buffer[random_task_file].split("\n") if line.startswith('class')]
            task_name = class_def[0]
            task_name = task_name[task_name.find("class "): task_name.rfind("(Task)")][6:]
            self.curr_task_name = self.generated_task_name = task_name

            self.generated_code = online_code_buffer[random_task_file]
            print("generated_code\n", self.generated_code)
            print("curr_task_name\n", self.curr_task_name)
            yield "Tarefa gerada ==> Asset gerado ==> API revisada ==> Erros revisados ==> Codigo gerado ==> ", "", self.generated_code, None

            self.generated_tasks.append(self.generated_task)
            self.generated_task_assets.append(self.generated_asset)
            self.generated_task_programs.append(self.generated_code)
            self.generated_task_names.append(self.generated_task_name)
        except:
            to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), HtmlFormatter())
            print("Task Creation Exception:", highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter()))
            self.log = to_print
            yield "Tarefa gerada ==> Asset gerado ==> API revisada ==> Erros revisados ==> Falha na geracao de codigo", self.log, "", None
            self.task_creation_pass = False
            return

        # self.curr_task_name = self.generated_task['task-name']
        print("task creation time {:.3f}".format(time.time() - start_time))

    def task_creation(self):
        """ create the task through interactions of agent and critic """
        self.task_creation_pass = True
        mkdir_if_missing(self.cfg['model_output_dir'])

        try:
            start_time = time.time()
            self.generated_task = self.agent.propose_task(self.generated_task_names)

            # self.generated_task = {'task-name': 'TASK_NAME_TEMPLATE', 'task-description': 'TASK_STRING_TEMPLATE', 'assets-used': ['ASSET_1', 'ASSET_2', Ellipsis]}
            print("generated_task\n", self.generated_task)
            yield "Tarefa gerada ==>", "", None, None
            self.generated_asset = self.agent.propose_assets()
            print("generated_asset\n", self.generated_asset)
            yield "Tarefa gerada ==> Asset gerado ==> ", "", None, None
            self.agent.api_review()
            yield "Tarefa gerada ==> Asset gerado ==> API revisada ==> ", "", None, None
            self.critic.error_review(self.generated_task)
            yield "Tarefa gerada ==> Asset gerado ==> API revisada ==> Erros revisados ==> ", "", None, None

            max_pose_retries = self.pose_validation_max_retries if self.pose_validation_enabled else 0
            validation_attempts = 0
            pose_status = "Tarefa gerada ==> Asset gerado ==> API revisada ==> Erros revisados ==> Validacao de pose falhou"

            while True:
                self.generated_code, self.curr_task_name = self.agent.implement_task()
                is_valid, reason = self._validate_generated_code(self.generated_code)
                if is_valid:
                    break

                validation_attempts += 1
                reason_html = self._format_html_traceback(reason)
                self.log = reason_html
                print("Pose validation failed:", reason)
                yield pose_status + f" ==> Nova tentativa {validation_attempts}", reason_html, self.generated_code, None

                if validation_attempts > max_pose_retries:
                    raise RuntimeError(reason)

            self.log = ""
            self.task_asset_logs.append(self.generated_task["assets-used"])
            self.generated_task_name = self.generated_task["task-name"]
            print("generated_code\n", self.generated_code)
            print("curr_task_name\n", self.curr_task_name)
            yield "Tarefa gerada ==> Asset gerado ==> API revisada ==> Erros revisados ==> Codigo gerado ==> ", self.log, self.generated_code, None

            self.generated_tasks.append(self.generated_task)
            self.generated_task_assets.append(self.generated_asset)
            self.generated_task_programs.append(self.generated_code)
            self.generated_task_names.append(self.generated_task_name)
        except:
            to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), HtmlFormatter())
            print("Task Creation Exception:", highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter()))
            self.log = to_print
            yield "Tarefa gerada ==> Asset gerado ==> API revisada ==> Erros revisados ==> Falha na geracao de codigo", self.log, "", None
            self.task_creation_pass = False
            return

        # self.curr_task_name = self.generated_task['task-name']
        print("task creation time {:.3f}".format(time.time() - start_time))


    def setup_env(self):
        """ build the new task"""
        env = Environment(
                self.cfg['assets_root'],
                disp=self.cfg['disp'],
                shared_memory=self.cfg['shared_memory'],
                hz=480,
                record_cfg=self.cfg['record']
            )

        task = eval(self.curr_task_name)()
        task.mode = self.cfg['mode']
        record = self.cfg['record']['save_video']
        save_data = self.cfg['save_data']

        # Initialize scripted oracle agent and dataset.
        expert = task.oracle(env)
        self.cfg['task'] = self.generated_task["task-name"]
        data_path = os.path.join(self.cfg['data_dir'], "{}-{}".format(self.generated_task["task-name"], task.mode))
        dataset = RavensDataset(data_path, self.cfg, n_demos=0, augment=False)
        print(f"Saving to: {data_path}")
        print(f"Mode: {task.mode}")

        # Start video recording
        if record:
            env.start_rec(f'{dataset.n_episodes+1:06d}')

        return task, dataset, env, expert

    def run_one_episode(self, dataset, expert, env, task, episode, seed):
        """ run the new task for one episode """
        add_to_txt(
                self.chat_log, f"================= TRIAL: {self.curr_trials}", with_print=True)
        record = self.cfg['record']['save_video']
        np.random.seed(seed)
        random.seed(seed)
        print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, self.cfg['n'], seed))
        env.set_task(task)
        obs = env.reset()

        info = env.info
        reward = 0
        total_reward = 0

        # Rollout expert policy
        for _ in range(task.max_steps):
            act = expert.act(obs, info)
            episode.append((obs, act, reward, info))
            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break

        episode.append((obs, None, reward, info))
        return total_reward

    def simulate_task(self):
        """ simulate the created task and save demonstrations """
        total_cnt = 0.
        reset_success_cnt = 0.
        env_success_cnt = 0.
        seed = 123
        self.curr_trials += 1
        
        if p.isConnected():
            p.disconnect()
        
        if not self.task_creation_pass:
            print("task creation failure => count as syntax exceptions.")
            return
        fix_attempts_used = 0
        status_prefix = "Tarefa gerada ==> Asset gerado ==> API revisada ==> Erros revisados ==> Codigo gerado ==> "

        while True:
            env = None
            try:
                exec(self.generated_code, globals())
                task, dataset, env, expert = self.setup_env()
                self.syntax_pass_rate += 1
            except Exception:
                trace_str = traceback.format_exc()
                html_trace = self._format_html_traceback(trace_str)
                save_text(self.cfg['model_output_dir'], self.generated_task_name + '_error', trace_str)
                print("========================================================")
                print("Syntax Exception:", highlight(f"{trace_str}", PythonLexer(), TerminalFormatter()))
                self.log = html_trace

                if self._can_attempt_fix(fix_attempts_used) and self._apply_auto_fix(trace_str, fix_attempts_used):
                    fix_attempts_used += 1
                    yield status_prefix + f"Falha de sintaxe do codigo ==> Tentativa automatica {fix_attempts_used}", self.log, self.generated_code, None
                    self._reset_physics()
                    continue

                yield status_prefix + "Falha de sintaxe do codigo", self.log, self.generated_code, None
                return

            try:
                env.generated_code = self.generated_code
                episode = []

                add_to_txt(
                    self.chat_log, f"================= TRIAL: {self.curr_trials}", with_print=True)
                np.random.seed(seed)
                random.seed(seed)
                print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, self.cfg['n'], seed))
                env.set_task(task)
                obs = env.reset()

                info = env.info
                reward = 0
                total_reward = 0

                start_time = time.time()
                print("start sim")
                for i in range(task.max_steps):
                    act = expert.act(obs, info)
                    episode.append((obs, act, reward, info))
                    lang_goal = info['lang_goal']

                    env.step(act)

                    obs, reward, done, info = env.cur_obs, env.cur_reward, env.cur_done, env.cur_info
                    total_reward += reward
                    print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')

                    if done:
                        break

                end_time = time.time()
                print("end sim, time used = ", end_time - start_time)

                if not os.path.exists(env.record_cfg['save_video_path']):
                    os.mkdir(env.record_cfg['save_video_path'])
                self.video_path = os.path.join(env.record_cfg['save_video_path'], "123.mp4")
                video_writer = imageio.get_writer(self.video_path,
                                                  fps=env.record_cfg['fps'],
                                                  format='FFMPEG',
                                                  codec='h264', )
                print(f"has {len(env.curr_video)} frames to save")
                for color in env.curr_video:
                    video_writer.append_data(color)
                video_writer.close()
                print("save video to ", self.video_path)

                yield status_prefix + "Simulacao concluida com sucesso", self.log, self.generated_code, self.video_path
                episode.append((obs, None, reward, info))
                print("Runtime Test Pass!")
                break
            except Exception:
                trace_str = traceback.format_exc()
                html_trace = self._format_html_traceback(trace_str)
                save_text(self.cfg['model_output_dir'], self.generated_task_name + '_error', trace_str)
                print("========================================================")
                print("Runtime Exception:", highlight(f"{trace_str}", PythonLexer(), TerminalFormatter()))
                self.log = html_trace

                if self._can_attempt_fix(fix_attempts_used) and self._apply_auto_fix(trace_str, fix_attempts_used):
                    fix_attempts_used += 1
                    yield status_prefix + f"Falha na execucao da simulacao ==> Tentativa automatica {fix_attempts_used}", self.log, self.generated_code, None
                    self._reset_physics()
                    continue

                yield status_prefix + "Falha na execucao da simulacao", self.log, self.generated_code, None
                return

        self.memory.save_run(self.generated_task)
