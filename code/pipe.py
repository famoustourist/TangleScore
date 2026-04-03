import json
import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from compute_z import compute_z
from util import nethook

import torch.optim as optim
from tqdm import tqdm
import argparse
import random
import numpy as np
import os
from transformers.modeling_attn_mask_utils import AttentionMaskConverter,_prepare_4d_causal_attention_mask

import torch.nn.functional as F

from geomloss import SamplesLoss


class UMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        diff = torch.abs(input - target)
        loss = (1 - diff).clamp(min=0)**2
        return loss.mean()


class AdaptiveUnlearningLoss(nn.Module):
    def __init__(self, margin=0.1, initial_weight=0.001, dynamic=True, total_steps=100):
        super().__init__()
        self.margin = margin
        self.initial_weight = initial_weight
        self.dynamic = dynamic
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        if self.dynamic:
            self.current_step += 1

    def compute_weight(self):
        return self.initial_weight

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        margin_mask = (diff < self.margin).float()
        loss = ((1 - diff).clamp(min=0))**2
        return -self.initial_weight * (loss * margin_mask).mean()


@torch.no_grad()
def compute_tangle_score(model, tok, question, answer, device):

    inputs = tok(question, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    r_old = outputs.hidden_states[-1].mean(dim=1).squeeze(0)

    new_input = question + " " + answer
    inputs_new = tok(new_input, return_tensors="pt").to(device)
    outputs_new = model(**inputs_new, output_hidden_states=True)
    r_new = outputs_new.hidden_states[-1].mean(dim=1).squeeze(0)

    D_semantic = 1 - torch.dot(r_old, r_new) / (
        torch.norm(r_old) * torch.norm(r_new) + 1e-8
    )

    token_1 = tok(answer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    token_2 = tok(question, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    embd_1 = model.model.embed_tokens(token_1).squeeze(0)
    embd_2 = model.model.embed_tokens(token_2).squeeze(0)

    prob_A = torch.ones(embd_1.size(0), device=device) / embd_1.size(0)
    prob_B = torch.ones(embd_2.size(0), device=device) / embd_2.size(0)

    ot_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    sinkhorn_dist = ot_loss(prob_A, embd_1, prob_B, embd_2)

    return D_semantic / (sinkhorn_dist + 1e-8)


def compute_alpha(ts):
    return torch.sigmoid(ts)


def compute_purge_rate(ts, lambda_=0.1, gamma=1.0):
    return lambda_ * (ts ** gamma)


def set_seed(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_ks(model, tok, batch_data, config, layer):
    input_ids = tok(batch_data, padding=True,return_tensors="pt").to(f"cuda:{config.device}")
    idxs = [i.sum()-1 for i in input_ids['attention_mask']]

    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=config.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
        ) as tr:
            _ = model(**input_ids)
            zs_out = tr.output

    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    zs_out = torch.stack([zs_out[i, idxs[i]] for i in range(len(zs_out))], dim=0)

    return zs_out, idxs


def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
    return [{'params': model.parameters(), 'lr': encoder_lr}]


def execute_batch_unke(model, tok, config, batch_data, ex_data):

    device = next(model.parameters()).device

    tangle_scores = []
    for data in batch_data:
        ts = compute_tangle_score(model, tok, data['question'], data['answer'], device)
        tangle_scores.append(ts)
    tangle_scores = torch.stack(tangle_scores)

    preserve_params = []
    for name, params in model.named_parameters():
        splitted_name = name.split('.')
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in config.layers:
                preserve_params.append(name)

    weights = {param: nethook.get_parameter(model, param) for param in preserve_params}
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    z_layer = config.layers[-1]
    z_list = []

    for data in batch_data:
        z_list.append(compute_z(model, tok, data, z_layer, config))

    zs = torch.stack(z_list, dim=0)
    batch_question = [i['question'] for i in batch_data]

    for i_layer, layer in enumerate(config.layers):

        contexts_tok = tok(batch_question, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=config.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input
                layer_out_ks = tr.output

        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks

        cur_zs, idxs = compute_ks(model, tok, batch_question, config, z_layer)
        targets = zs - cur_zs
        resid = targets / (len(config.layers) - i_layer)

        for i in range(len(idxs)):
            layer_out_ks[i, idxs[i]] += resid[i]

        ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=config.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**ex_tok)
                stat_in = tr.input
                stat_out = tr.output

        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out

        _layer = nethook.get_module(model, config.layer_module_tmp.format(layer))
        optimizer = optim.AdamW(_layer.parameters(), lr=config.lr)
        criterion = nn.MSELoss()

        pre_layer_out_ks = layer_out_ks.clone().detach()

        for step in range(config.optim_num_step):
            optimizer.zero_grad()

            # ===== PIPE替换loss =====
            for i in range(len(batch_data)):

                ts_i = tangle_scores[i]
                alpha = compute_alpha(ts_i)
                PR = compute_purge_rate(ts_i)

                pred_edit = _layer(layer_in_ks)[0]
                pred_preserve = _layer(stat_in)[0]

                loss_edit = criterion(pred_edit, layer_out_ks)
                loss_preserve = criterion(pred_preserve, stat_out)

                loss_align = alpha * loss_edit + (1 - alpha) * loss_preserve

                loss_unlearn = PR * (
                    (1 - torch.abs(pred_edit - pre_layer_out_ks)).clamp(min=0)**2
                ).mean()

                loss = loss_align + loss_unlearn
                loss.backward(retain_graph=True)

            optimizer.step()

    return weights_copy


def get_llama_without_answer(que):
    return f"""<s>[INST] {que} [/INST]"""


def get_list_llama_without_answer(que, cot):
    return [get_llama_without_answer(line) for line in que]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":

    set_seed()
    config = Config()

    args = parse_args()
    data_path = args.data_path or config.data_path
    output_path = args.output_path or '../output/test_2.json'

    with open(data_path, 'r', encoding='utf-8') as json_file:
        edit_data = json.load(json_file)

    with open(config.ex_data_path, 'r', encoding='utf-8') as json_file:
        ex_datas = json.load(json_file)

    ex_datas = [i['instruction'] + i['input'] + i['output'] for i in ex_datas]

    model = AutoModelForCausalLM.from_pretrained(config.model_path, device_map=f"cuda:{config.device}")
    tok = AutoTokenizer.from_pretrained(config.model_path)

    batch_size = config.batch_size
    num_batches = len(edit_data) // batch_size + (1 if len(edit_data) % batch_size else 0)

    edited_data = []

    for batch_index in tqdm(range(num_batches)):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        batch = edit_data[start_index:end_index]

        random_elements = random.sample(ex_datas, config.ex_data_num)
        weights_copy = execute_batch_unke(model, tok, config, batch, random_elements)

        edited_data.extend(batch)

        if config.keep_original_weight:
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to(f"cuda:{config.device}")

    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(edited_data, json_file, ensure_ascii=False, indent=4)

    print(f"saving to {output_path}")