import os
import sys
import json

import numpy as np
from cliport import tasks
from cliport import agents
from cliport.utils import utils

import torch
import cv2
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
from torch.utils.data import DataLoader
import IPython

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


train_demos = 50 # number training demonstrations used to train agent
n_eval = 1 # number of evaluation instances
mode = 'test' # val or test

agent_name = 'cliport'
model_task = 'place-red-in-green' # multi-task agent conditioned with language goals
task_type = 'cliport3_task_indomain'  # cliport3_task_indomain, gpt5_mixcliport2
# model_folder = f'exps/exp-{task_type}_demo{train_demos}_2023-07-27_13-30-52-small' # path to pre-trained checkpoint

# Lirui
model_folder = f'exps-singletask/debug_checkpoints' # path to pre-trained checkpoint
ckpt_name = 'last.ckpt' # name of checkpoint to load

draw_grasp_lines = True
affordance_heatmap_scale = 30

### Uncomment the task you want to evaluate on ###
# eval_task = 'align-rope'
# eval_task = 'assembling-kits-seq-seen-colors'
# eval_task = 'assembling-kits-seq-unseen-colors'
# eval_task = 'packing-shapes'
# eval_task = 'packing-boxes-pairs-seen-colors'
# eval_task = 'packing-boxes-pairs-unseen-colors'
# eval_task = 'packing-seen-google-objects-seq'
# eval_task = 'packing-unseen-google-objects-seq'
# eval_task = 'packing-seen-google-objects-group'
# eval_task = 'packing-unseen-google-objects-group'
# eval_task = 'put-block-in-bowl-seen-colors'
# eval_task = 'put-block-in-bowl-unseen-colors'
eval_task = 'place-red-in-green'
# eval_task = 'stack-block-pyramid-seq-unseen-colors'
# eval_task = 'separating-piles-seen-colors'
# eval_task = 'separating-piles-unseen-colors'
# eval_task = 'towers-of-hanoi-seq-seen-colors'
# eval_task = 'towers-of-hanoi-seq-unseen-colors'



def crop_img(img, height_range=[200, 340], width_range=[180, 460]):
    img = img[height_range[0]:height_range[1], width_range[0]:width_range[1], :]
    return img

def read_rgb_image(path):
    img = cv2.imread(path)
    img = crop_img(img)
    img = cv2.resize(img, (320, 160))
    img = img.transpose(1, 0, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_depth_image(path):
    # TODO: why the depth image has 4 channels ?
    img = plt.imread(path, cv2.IMREAD_UNCHANGED) # TODO: need correct
    img = crop_img(img)
    img = cv2.resize(img, (320, 160))[:, :, 0][:, :, None]
    img = img.transpose(1, 0, 2)
    return img

def process_real_sample(cmap, dmap, info, aug_theta_sigma=60, augment=False):
    """Process the sample like the dataset method."""
    print(cmap.shape, dmap.shape)
    img = np.concatenate((cmap, dmap, dmap, dmap), axis=2)
    p0, p1 = np.zeros(1), np.zeros(1)
    p0_theta, p1_theta = np.zeros(1), np.zeros(1)
    perturb_params =  np.zeros(5)
    if augment:
        img, _, (p0, p1), perturb_params = utils.perturb(img, [p0, p1], theta_sigma=aug_theta_sigma)
    
    sample = {
            'img': img.copy(),
            'p0': np.array(p0).copy(), 'p0_theta': np.array(p0_theta).copy(),
            'p1': np.array(p1).copy(), 'p1_theta': np.array(p1_theta).copy() ,
            'perturb_params': np.array(perturb_params).copy() 
        }
    
    if info and 'lang_goal' in info:
        sample['lang_goal'] = info['lang_goal']
    
    return sample
    

def plot_affordance(batch, obs, agent, info, draw_grasp_lines=True, affordance_heatmap_scale=30):

    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
            
    # Get color and depth inputs
    img = batch['img']  # (320, 160, 6)
    img = torch.from_numpy(img)
    color = np.uint8(img.detach().cpu().numpy())[:,:,:3]
    color = color.transpose(1,0,2)
    depth = np.array(img.detach().cpu().numpy())[:,:,3]
    depth = depth.transpose(1,0)

    # Display input color
    axs[0,0].imshow(color)
    axs[0,0].axes.xaxis.set_visible(False)
    axs[0,0].axes.yaxis.set_visible(False)
    axs[0,0].set_title('Input RGB')

    # Display input depth
    axs[0,1].imshow(depth)
    axs[0,1].axes.xaxis.set_visible(False)
    axs[0,1].axes.yaxis.set_visible(False)        
    axs[0,1].set_title('Input Depth')

    # Display predicted pick affordance
    axs[1,0].imshow(color)
    axs[1,0].axes.xaxis.set_visible(False)
    axs[1,0].axes.yaxis.set_visible(False)
    axs[1,0].set_title('Pick Affordance')

    # Display predicted place affordance
    axs[1,1].imshow(color)
    axs[1,1].axes.xaxis.set_visible(False)
    axs[1,1].axes.yaxis.set_visible(False)
    axs[1,1].set_title('Place Affordance')

    # Get action predictions
    l = str(info['lang_goal'])
    act = agent.real_act(obs, info, goal=None)
    pick, place = act['pick'], act['place']

    # Visualize pick affordance
    pick_inp = {'inp_img': batch['img'], 'lang_goal': l}
    pick_conf = agent.attn_forward(pick_inp)[0]
    print("pick_conf:", pick_conf.shape, pick, place)
    # IPython.embed()
    logits = pick_conf.detach().cpu().numpy()

    pick_conf = pick_conf.detach().cpu().numpy()
    argmax = np.argmax(pick_conf)
    argmax = np.unravel_index(argmax, shape=pick_conf.shape)
    p0 = argmax[:2]

    p0_theta = (argmax[2] * (2 * np.pi / pick_conf.shape[2])) * -1.0

    line_len = 30
    pick0 = (pick[0] + line_len/2.0 * np.sin(p0_theta), pick[1] + line_len/2.0 * np.cos(p0_theta))
    pick1 = (pick[0] - line_len/2.0 * np.sin(p0_theta), pick[1] - line_len/2.0 * np.cos(p0_theta))

    if draw_grasp_lines:
        axs[1,0].plot((pick1[0], pick0[0]), (pick1[1], pick0[1]), color='r', linewidth=1)

    # Visualize place affordance
    place_inp = {'inp_img': batch['img'], 'p0': pick, 'lang_goal': l}
    place_conf = agent.trans_forward(place_inp)[0]

    place_conf = place_conf.permute(1, 2, 0)
    place_conf = place_conf.detach().cpu().numpy()
    argmax = np.argmax(place_conf)
    argmax = np.unravel_index(argmax, shape=place_conf.shape)
    p1_pix = argmax[:2]
    p1_theta = (argmax[2] * (2 * np.pi / place_conf.shape[2]) + p0_theta) * -1.0

    line_len = 30
    place0 = (place[0] + line_len/2.0 * np.sin(p1_theta), place[1] + line_len/2.0 * np.cos(p1_theta))
    place1 = (place[0] - line_len/2.0 * np.sin(p1_theta), place[1] - line_len/2.0 * np.cos(p1_theta))

    if draw_grasp_lines:
        axs[1,1].plot((place1[0], place0[0]), (place1[1], place0[1]), color='g', linewidth=1)

    # Overlay affordances on RGB input
    pick_logits_disp = np.uint8(logits * 255 * affordance_heatmap_scale).transpose(2,1,0)
    place_logits_disp = np.uint8(np.sum(place_conf, axis=2)[:,:,None] * 255 * affordance_heatmap_scale).transpose(1,0,2)# .transpose(1,2,0)    

    pick_logits_disp_masked = np.ma.masked_where(pick_logits_disp < 0, pick_logits_disp)
    place_logits_disp_masked = np.ma.masked_where(place_logits_disp < 0, place_logits_disp)
    # IPython.embed()

    axs[1][0].imshow(pick_logits_disp_masked, alpha=0.75)
    axs[1][1].imshow(place_logits_disp_masked, cmap='viridis', alpha=0.75)

    print(f"Lang Goal: {str(info['lang_goal'])}")
    plt.savefig(f'{root_dir}/data/real_output/test_real_affordance2.png')


if __name__ == '__main__':
    os.environ['GENSIM_ROOT'] = f'{os.path.abspath(__file__)}/../..'
    root_dir = os.environ['GENSIM_ROOT']
    print("root_dir:", root_dir)
    assets_root = os.path.join(root_dir, 'cliport/environments/assets/')
    config_file = 'eval.yaml'

    vcfg = utils.load_hydra_config(os.path.join(root_dir, f'cliport/cfg/{config_file}'))
    vcfg['data_dir'] = os.path.join(root_dir, 'data')
    vcfg['mode'] = mode

    vcfg['model_task'] = model_task
    vcfg['eval_task'] = eval_task
    vcfg['agent'] = agent_name

    # Model and training config paths
    model_path = os.path.join(root_dir, model_folder)
    if model_folder[-7:] == 'smaller':
        vcfg['train_config'] = f"{model_path}/{model_folder[9:-8]}-{vcfg['agent']}-n{train_demos}-train/.hydra/config.yaml"
        vcfg['model_path'] = f"{model_path}/{model_folder[9:-8]}-{vcfg['agent']}-n{train_demos}-train/checkpoints/"
    else:
        vcfg['train_config'] = f"{model_path}/{model_folder[9:-6]}-{vcfg['agent']}-n{train_demos}-train/.hydra/config.yaml"
        vcfg['model_path'] = f"{model_path}/{model_folder[9:-6]}-{vcfg['agent']}-n{train_demos}-train/checkpoints/"
    tcfg = utils.load_hydra_config(vcfg['train_config'])

    # Load dataset
    ds = RavensDataset(os.path.join(vcfg['data_dir'], f'{vcfg["eval_task"]}-{vcfg["mode"]}'),
                       tcfg,
                       n_demos=n_eval,
                       augment=False)

    eval_run = 0
    name = '{}-{}-{}-{}'.format(vcfg['eval_task'], vcfg['agent'], n_eval, eval_run)
    print(f'\nEval ID: {name}\n')

    # Initialize agent
    utils.set_seed(eval_run, torch=True)
    agent = agents.names[vcfg['agent']](name, tcfg, DataLoader(ds), DataLoader(ds))

    # Load checkpoint
    ckpt_path = os.path.join(vcfg['model_path'], ckpt_name)
    print(f'\nLoading checkpoint: {ckpt_path}')
    agent.load(ckpt_path)

    os.makedirs(f'{root_dir}/data/real_output', exist_ok=True)
    real_rgb_img = read_rgb_image(f'{root_dir}/data/real_imgs/rgb0.png')
    plt.imshow(real_rgb_img[:, :, :3])
    plt.axis('off')
    plt.savefig(f'{root_dir}/data/real_output/real_show.png')
    real_depth_img = read_depth_image(f'{root_dir}/data/real_imgs/depth0.png')
    print(real_depth_img.shape, real_rgb_img.shape)
    plt.imshow(real_depth_img, cmap='gray')
    plt.savefig(f'{root_dir}/data/real_output/real_depth.png')
    info = {}
    info['lang_goal'] = 'place red block in green bowl'

    batch = process_real_sample(real_rgb_img, real_depth_img, info, augment=False)

    obs = batch['img']
    plot_affordance(batch, obs, agent, info, draw_grasp_lines=draw_grasp_lines, affordance_heatmap_scale=affordance_heatmap_scale)