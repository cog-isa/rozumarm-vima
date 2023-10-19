import os

import numpy as np
from tokenizers import Tokenizer
from tokenizers import AddedToken
from einops import rearrange
import cv2
from vima.utils import *
from vima import create_policy_from_ckpt
from vima_bench import *
from gym.wrappers import TimeLimit as _TimeLimit
from gym import Wrapper
import torch
import argparse
import matplotlib.pyplot as plt
import copy
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import vimasim
import torch
import sys
sys.path.insert(0, './src_rudolph/')
import rudolph
from rudolph.model import get_rudolph_model, ruDolphModel, FP16Module
from rudalle import get_tokenizer, get_vae
from rudolph.api import ruDolphApi
from rudolph.model.utils import get_attention_mask


os.environ["TOKENIZERS_PARALLELISM"] = "true"
_kwargs = {"single_word": True,    "lstrip": False,    "rstrip": False,    "normalized": True,}

def de_discretize_actions( actions):
    actions = {k: v.float() for k, v in actions.items()}
    actions["pose0_position"][..., 0] = (
        actions["pose0_position"][..., 0] / 50
    )
    actions["pose0_position"][..., 1] = (
        actions["pose0_position"][..., 1] / 100
    )
    actions["pose0_rotation"] = (
        actions["pose0_rotation"] / 50
    )

    actions["pose1_position"][..., 0] = (
        actions["pose1_position"][..., 0] / 50
    )
    actions["pose1_position"][..., 1] = (
        actions["pose1_position"][..., 1] / 100
    )
    actions["pose1_rotation"] = (
        actions["pose1_rotation"] / 50
    )
    return actions

class RuDolphModel:

    def __init__(self, use_mock_api=False):

        self.SPC_TOKENS = {
            '<LT_UNK>': 16384,
            '<RT_UNK>': 16385,
            '<LT_T2I>': 16386,
            '<LT_I2T>': 16387,
            '<LT_T2T>': 16388,
            '<RT_I2T>': 16389,
            
            '<LT_TQA>': 16390,
            '<RT_TQA>': 16391,
            
            '<LT_RLA>': 16392,
            '<RT_RLA>': 16393,
        }
        for i in range(50):
            self.SPC_TOKENS['VIMA_X_'+str(i)] = 16394 + i

        for i in range(100):
            self.SPC_TOKENS['VIMA_Y_'+str(i)] = 16394 + 50 + i
            
        for i in range(50):
            self.SPC_TOKENS['VIMA_ROT_'+str(i)] = 16394 + 50 + 100 + i    

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = get_rudolph_model('350M', pretrained=True, fp16=True, device=self.device) #2.7B #1.3B #350M
        checkpoint_path = '/home/daniil/code/rozumarm-vima/VIMA_SWEEP.pt'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.tokenizer_rudolph = get_tokenizer()
        self.vae = get_vae(dwt=False).to(self.device)
        self.api = ruDolphApi(self.model, self.tokenizer_rudolph, self.vae, bs=48)

        class Args():
            def __init__(self, model, checkpoint_path):
                self.device = model.get_param('device')
                self.l_text_seq_length = model.get_param('l_text_seq_length')
                self.r_text_seq_length = model.get_param('r_text_seq_length')
                self.image_tokens_per_dim = model.get_param('image_tokens_per_dim')
                self.image_seq_length = model.get_param('image_seq_length')
                self.epochs = 2
                self.save_path= checkpoint_path
                self.model_name = 'rudolph_sberquad_'
                self.save_every = 500
                self.bs = 8
                self.clip = 1.0
                self.lr = 2e-5
                self.wandb = False
                self.lt_loss_weight = 0.0
                self.img_loss_weight = 0.0
                self.rt_loss_weight = 7
                self.image_size = self.image_tokens_per_dim * 8
                
        checkpoint_path = '/home/daniil/code/rozumarm-vima-utils/src_rudolph/'
        self.args = Args(self.model, checkpoint_path)

    def reset(self, prompt, prompt_assets):
        self.prompt = prompt
        self.prompt_assets = prompt_assets
        self.elapsed_steps = 0
        return None
    
    def step(self,obs,meta_info):
        self.elapsed_steps +=1
        left_special_token = '<LT_RLA>'
        right_special_token = '<RT_RLA>'

        lt = torch.zeros(self.args.l_text_seq_length,dtype=torch.int32)
        lt[0] = 2
        lt[1] = self.SPC_TOKENS[left_special_token]
        lt[2:] = self.tokenizer_rudolph.encode_text(self.prompt.lower().strip().replace('{',''), text_seq_length=self.args.l_text_seq_length)[1:-1]

        rt = torch.zeros(2, dtype=torch.int32)
        rt[0] = 2
        rt[1] = self.SPC_TOKENS[right_special_token]

        image_step_top = obs['rgb']['top'].transpose(1,2,0)
        img = np.vstack((np.hstack((self.prompt_assets['bounds']['rgb']['top'].transpose(1,2,0)[:,64:-64,:] +
        self.prompt_assets['constraint']['rgb']['top'].transpose(1,2,0)[:,64:-64,:],
                    self.prompt_assets['swept_obj']['rgb']['top'].transpose(1,2,0)[:,64:-64,:])),
                    image_step_top))
        img = Image.fromarray(img)
        img = self.api.image_transform(img)
        img = img.unsqueeze(0).to(self.api.device)
        image_input_ids_text = self.api.vae.get_codebook_indices(img, disable_gumbel_softmax=True)[0]

        attention_mask_text = get_attention_mask(1, self.args.l_text_seq_length,self.args.image_tokens_per_dim,self.args.r_text_seq_length, self.args.device)
        input_ids_text = torch.cat((lt.to(self.args.device).unsqueeze(0), image_input_ids_text.to(self.args.device).unsqueeze(0), rt.to(self.args.device).unsqueeze(0)), dim=1)
        spcs = [['VIMA_X_0',50],['VIMA_Y_0',100],['VIMA_ROT_0',50],['VIMA_ROT_0',50],['VIMA_ROT_0',50],['VIMA_ROT_0',50]]
        spcs*=2
        actionss = []
        for i in range(12):
            with torch.no_grad():
                logits = self.model(input_ids_text, attention_mask_text)
                print(logits[0].shape)
            #a_t = torch.argmax(logits[0][:, -1, SPC_TOKENS[spcs[i][0]]:SPC_TOKENS[spcs[i][0]]+spcs[i][1]]).item()
            distribution = torch.softmax(logits[0][:, -1, self.SPC_TOKENS[spcs[i][0]]:self.SPC_TOKENS[spcs[i][0]]+spcs[i][1]], 1)
            a_t = torch.multinomial(distribution, 1).item()
            actionss.append(a_t)
            input_ids_text = torch.cat((input_ids_text,torch.tensor([[self.SPC_TOKENS[spcs[i][0]]+a_t]]).to(self.device)), dim=1)  
        actions = {'pose0_position': torch.tensor([[[actionss[0], actionss[1]]]], device='cuda:0'), 
                'pose0_rotation': torch.tensor([[[actionss[2], actionss[3], actionss[4], actionss[5]]]], device='cuda:0'), 
                'pose1_position': torch.tensor([[[actionss[6], actionss[7]]]], device='cuda:0'), 
                'pose1_rotation': torch.tensor([[[actionss[8], actionss[8], actionss[10], actionss[11]]]], device='cuda:0')}  

        print(actions)
        #action_tokens = policy.forward_action_token(actions)  # (1, B, E)
        #action_tokens = action_tokens.squeeze(0)  # (B, E)
        #inference_cache["action_tokens"].append(action_tokens[0])
        actions = de_discretize_actions(actions)
        action_bounds = [meta_info["action_bounds"]]
        action_bounds_low = [action_bound["low"] for action_bound in action_bounds]
        action_bounds_high = [
            action_bound["high"] for action_bound in action_bounds
        ]
        action_bounds_low = np.asarray(action_bounds_low)
        action_bounds_high = np.asarray(action_bounds_high)
        action_bounds_low = torch.tensor(
            action_bounds_low, dtype=torch.float32, device=self.api.device
        )
        action_bounds_high = torch.tensor(
            action_bounds_high, dtype=torch.float32, device=self.api.device
        )
        actions["pose0_position"] = (
            actions["pose0_position"] * (action_bounds_high - action_bounds_low)
            + action_bounds_low
        )
        actions["pose1_position"] = (
            actions["pose1_position"] * (action_bounds_high - action_bounds_low)
            + action_bounds_low
        )
        actions["pose0_position"] = torch.clamp(
            actions["pose0_position"], min=action_bounds_low, max=action_bounds_high
        )
        actions["pose1_position"] = torch.clamp(
            actions["pose1_position"], min=action_bounds_low, max=action_bounds_high
        )
        actions["pose0_rotation"] = actions["pose0_rotation"] * 2 - 1
        actions["pose1_rotation"] = actions["pose1_rotation"] * 2 - 1
        actions["pose0_rotation"] = torch.clamp(
            actions["pose0_rotation"], min=-1, max=1
        )
        actions["pose1_rotation"] = torch.clamp(
            actions["pose1_rotation"], min=-1, max=1

        )
        actions = {k: v.cpu().numpy() for k, v in actions.items()}
        actions = any_slice(actions, np.s_[0, 0])

        
        return actions



"""
torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_rudolph_model('350M', pretrained=True, fp16=True, device=device) #2.7B #1.3B #350M
checkpoint_path = '/home/daniil/code/rozumarm-vima-utils/VIMA_SWEEP.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
tokenizer_rudolph = get_tokenizer()
vae = get_vae(dwt=False).to(device)
api = ruDolphApi(model, tokenizer_rudolph, vae, bs=48)
class Args():
    def __init__(self, model, checkpoint_path):
        self.device = model.get_param('device')
        self.l_text_seq_length = model.get_param('l_text_seq_length')
        self.r_text_seq_length = model.get_param('r_text_seq_length')
        self.image_tokens_per_dim = model.get_param('image_tokens_per_dim')
        self.image_seq_length = model.get_param('image_seq_length')
        self.epochs = 2
        self.save_path= checkpoint_path
        self.model_name = 'rudolph_sberquad_'
        self.save_every = 500
        self.bs = 8
        self.clip = 1.0
        self.lr = 2e-5
        self.wandb = False
        self.lt_loss_weight = 0.0
        self.img_loss_weight = 0.0
        self.rt_loss_weight = 7
        self.image_size = self.image_tokens_per_dim * 8
        
checkpoint_path = '../../checkpoints/350_S3_FBC/'
args = Args(model, checkpoint_path)

"""