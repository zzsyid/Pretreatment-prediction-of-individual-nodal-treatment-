import gc
import json
import os
import random
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from ranking import supervised_risk_ranking_loss
MODALITY_ORDER = ('t1', 't1c', 't2', 'ct', 'pet')
MRI_MODALITIES = ('t1', 't1c', 't2')
CT_MODALITIES = ('ct', 'pet')

def apply_source_family_dropout(mri_present, ct_present):
    return (mri_present, ct_present)

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

def prepare_teacher_mri_volumes(volumes, mode='current_log'):
    if mode != 'current_log':
        raise ValueError('The final method uses log1p MRI-volume metadata')
    return volumes

def disable_inplace(model):
    for name, module in model.named_children():
        if hasattr(module, 'inplace'):
            module.inplace = False
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ReLU6)):
            setattr(model, name, type(module)(inplace=False))
        else:
            disable_inplace(module)
    return model

def load_checkpoint_state(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    cleaned_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=True)
    return model

def configure_cpea_stage2(model, zero_init_effect_head=False):
    required = ('pet_response_generator', 'pet_effect_adapter', 'feature_fusers')
    if not all((hasattr(model, name) for name in required)):
        raise ValueError('--freeze-student-base requires the dualfusion CPEA student model')
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_modules = (model.pet_response_generator, model.pet_effect_adapter, model.feature_fusers['pet'])
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.cpea_stage2_frozen = True
    if zero_init_effect_head:
        output_layer = model.pet_effect_adapter.delta_logit_head[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError('Expected the final PET effect head layer to be nn.Linear')
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]

def get_effective_weights(args):
    return {'struct_consistency_weight': 0.0 if args.disable_struct_loss else args.struct_consistency_weight, 'orthogonality_weight': 0.0 if args.disable_orth_loss else args.orthogonality_weight, 'response_diversity_weight': 0.0 if args.disable_diversity_loss else args.response_diversity_weight, 'kd_logits_weight': 0.0 if args.disable_kd_logits else args.kd_logits_weight, 'kd_feature_weight': 0.0 if args.disable_kd_token else args.kd_feature_weight, 'student_struct_kd_weight': 0.0 if args.disable_kd_struct else args.student_struct_kd_weight, 'student_response_kd_weight': 0.0 if args.disable_kd_response else args.student_response_kd_weight, 'student_base_cls_weight': 0.0 if args.disable_aux_outputs else args.student_base_cls_weight, 'student_pet_proxy_kd_weight': 0.0 if args.disable_kd_response else args.student_pet_proxy_kd_weight, 'student_pet_proxy_weight_kd_weight': 0.0 if args.disable_kd_response else args.student_pet_proxy_weight_kd_weight, 'student_pet_delta_kd_weight': 0.0 if args.disable_kd_response else args.student_pet_delta_kd_weight, 'student_pet_effect_kd_weight': 0.0 if args.disable_kd_logits else args.student_pet_effect_kd_weight, 'student_pet_effect_sign_weight': 0.0 if args.disable_kd_logits else args.student_pet_effect_sign_weight, 'student_pet_utility_weight': 0.0 if args.disable_kd_logits else args.student_pet_utility_weight, 'student_pet_residual_weight': 0.0 if args.disable_aux_outputs else args.student_pet_residual_weight, 'student_pet_decorrelation_weight': 0.0 if args.disable_aux_outputs else args.student_pet_decorrelation_weight, 'student_pet_gain_logit_kd_weight': 0.0 if args.disable_kd_logits else args.student_pet_gain_logit_kd_weight, 'student_pet_ranking_kd_weight': 0.0 if args.disable_kd_logits else args.student_pet_ranking_kd_weight, 'student_supervised_ranking_weight': getattr(args, 'student_supervised_ranking_weight', 0.0), 'generated_response_kd_weight': 0.0 if args.disable_kd_response else args.generated_response_kd_weight, 'meanflow_response_weight': 0.0 if args.disable_kd_response else args.meanflow_response_weight, 'generation_quality_weight': 0.0 if args.disable_kd_response else args.generation_quality_weight}

def get_student_hide_probs(args, epoch):
    max_t1c = args.student_hide_prob_t1c
    max_pet = args.student_hide_prob_pet
    if args.student_hide_schedule == 'fixed' or args.epochs <= 1:
        return (max_t1c, max_pet)
    progress = epoch / max(args.epochs - 1, 1)
    if args.student_hide_schedule == 'linear':
        scale = progress
    else:
        scale = 0.5 * (1.0 - np.cos(np.pi * progress))
    return (max_t1c * scale, max_pet * scale)

def apply_meta_ablation(vols, regions, disable_meta):
    if not disable_meta:
        return (vols, regions)
    return (torch.zeros_like(vols), torch.zeros_like(regions))

def _filter_json_items_by_stage(data_list, stage):
    if stage == 1:
        modalities = ['T1', 'T2', 'T1CE', 'CT', 'PET']
        return [item for item in data_list if all((item.get(m) is not None for m in modalities))]
    return data_list

def calculate_pos_weight_from_json(json_path, stage, extra_json_paths=None, extra_stage=2, extra_repeat=1):
    with open(json_path, 'r', encoding='utf-8') as file:
        data_list = _filter_json_items_by_stage(json.load(file), stage)
    if extra_json_paths:
        repeat = max(int(extra_repeat), 1)
        for extra_json_path in extra_json_paths:
            with open(extra_json_path, 'r', encoding='utf-8') as file:
                extra_items = _filter_json_items_by_stage(json.load(file), extra_stage)
            data_list.extend(extra_items * repeat)
    labels = [item.get('label', 0) for item in data_list]
    num_pos = sum((1 for label in labels if label == 1))
    num_neg = sum((1 for label in labels if label == 0))
    if num_pos == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([num_neg / num_pos], dtype=torch.float32)

def extract_batch(batch, device):
    mri_imgs = torch.cat([batch['T1'], batch['T1CE'], batch['T2']], dim=1).as_tensor().float().to(device).contiguous()
    ct_pet_imgs = torch.cat([batch['CT'], batch['PET']], dim=1).as_tensor().float().to(device).contiguous()
    batch_size = mri_imgs.size(0)
    mri_present = {'t1': batch['T1_present'].to(device).float().view(batch_size), 't1c': batch['T1CE_present'].to(device).float().view(batch_size), 't2': batch['T2_present'].to(device).float().view(batch_size)}
    ct_present = {'ct': batch['CT_present'].to(device).float().view(batch_size), 'pet': batch['PET_present'].to(device).float().view(batch_size)}
    vols = batch['mri_physical_volume'].to(device).float().contiguous()
    regions = batch['region'].to(device).float().contiguous()
    labels = batch['label'].to(device).float().contiguous()
    return (mri_imgs, ct_pet_imgs, mri_present, ct_present, vols, regions, labels)

def mask_images_by_present(mri_imgs, ct_pet_imgs, mri_present, ct_present):
    batch_size = mri_imgs.size(0)
    mri_mask = torch.stack([mri_present['t1'], mri_present['t1c'], mri_present['t2']], dim=1).view(batch_size, 3, 1, 1, 1)
    ct_mask = torch.stack([ct_present['ct'], ct_present['pet']], dim=1).view(batch_size, 2, 1, 1, 1)
    return (mri_imgs * mri_mask, ct_pet_imgs * ct_mask)

def apply_input_mode(mri_imgs, mri_present, ct_pet_imgs, ct_present, input_mode):
    if input_mode in {'limited', 'with_t1c', 'with_pet', 'mri_only', 'ct_only'}:
        mri_present = dict(mri_present)
        ct_present = dict(ct_present)
        if input_mode in {'limited', 'with_pet', 'mri_only', 'ct_only'}:
            mri_present['t1c'] = torch.zeros_like(mri_present['t1c'])
        if input_mode in {'limited', 'with_t1c', 'mri_only', 'ct_only'}:
            ct_present['pet'] = torch.zeros_like(ct_present['pet'])
        if input_mode == 'mri_only':
            ct_present['ct'] = torch.zeros_like(ct_present['ct'])
        if input_mode == 'ct_only':
            mri_present['t1'] = torch.zeros_like(mri_present['t1'])
            mri_present['t2'] = torch.zeros_like(mri_present['t2'])
    mri_imgs, ct_pet_imgs = mask_images_by_present(mri_imgs, ct_pet_imgs, mri_present, ct_present)
    return (mri_imgs, mri_present, ct_pet_imgs, ct_present)

def make_student_present_dict(real_mri_present, real_ct_present, hide_prob_t1c=1.0, hide_prob_pet=1.0):
    s_mri_present = {'t1': real_mri_present['t1'], 't2': real_mri_present['t2'], 't1c': real_mri_present['t1c'] * (torch.rand_like(real_mri_present['t1c']) > hide_prob_t1c).float()}
    s_ct_present = {'ct': real_ct_present['ct'], 'pet': real_ct_present['pet'] * (torch.rand_like(real_ct_present['pet']) > hide_prob_pet).float()}
    return (s_mri_present, s_ct_present)

def apply_student_input_mode(student_mri_present, student_ct_present, input_mode):
    if input_mode == 'limited':
        student_mri_present = dict(student_mri_present)
        student_ct_present = dict(student_ct_present)
        student_mri_present['t1c'] = torch.zeros_like(student_mri_present['t1c'])
        student_ct_present['pet'] = torch.zeros_like(student_ct_present['pet'])
    if input_mode == 'with_t1c':
        student_ct_present = dict(student_ct_present)
        student_ct_present['pet'] = torch.zeros_like(student_ct_present['pet'])
    if input_mode == 'with_pet':
        student_mri_present = dict(student_mri_present)
        student_mri_present['t1c'] = torch.zeros_like(student_mri_present['t1c'])
    if input_mode == 'mri_only':
        student_mri_present = dict(student_mri_present)
        student_ct_present = dict(student_ct_present)
        student_mri_present['t1c'] = torch.zeros_like(student_mri_present['t1c'])
        student_ct_present['ct'] = torch.zeros_like(student_ct_present['ct'])
        student_ct_present['pet'] = torch.zeros_like(student_ct_present['pet'])
    if input_mode == 'ct_only':
        student_mri_present = dict(student_mri_present)
        student_ct_present = dict(student_ct_present)
        student_mri_present['t1'] = torch.zeros_like(student_mri_present['t1'])
        student_mri_present['t1c'] = torch.zeros_like(student_mri_present['t1c'])
        student_mri_present['t2'] = torch.zeros_like(student_mri_present['t2'])
        student_ct_present['pet'] = torch.zeros_like(student_ct_present['pet'])
    return (student_mri_present, student_ct_present)

def get_full_modality_mask(mri_present, ct_present):
    return (mri_present['t1'] > 0.5) & (mri_present['t1c'] > 0.5) & (mri_present['t2'] > 0.5) & (ct_present['ct'] > 0.5) & (ct_present['pet'] > 0.5)

def masked_mse_loss(input_tensor, target_tensor, sample_mask, device):
    if sample_mask is None:
        return F.mse_loss(input_tensor, target_tensor.detach())
    if not sample_mask.any():
        return torch.tensor(0.0, device=device)
    return F.mse_loss(input_tensor[sample_mask], target_tensor.detach()[sample_mask])

def compute_alignment_loss(teacher_token, student_token):
    target = teacher_token.detach()
    cos_sim = F.cosine_similarity(student_token, target, dim=-1)
    return (1.0 - cos_sim).mean()

def get_model_aux(model):
    base_model = model.module if isinstance(model, DDP) else model
    return getattr(base_model, 'latest_aux', None)

def set_aux_outputs_enabled(model, enabled):
    base_model = model.module if isinstance(model, DDP) else model
    if hasattr(base_model, 'enable_aux_outputs'):
        base_model.enable_aux_outputs = enabled

def _append_masked_cosine_distance(losses, feat_a, feat_b, valid_mask, weight=1.0):
    if valid_mask.any():
        cos_sim = F.cosine_similarity(feat_a[valid_mask], feat_b[valid_mask], dim=-1)
        losses.append(weight * (1.0 - cos_sim).mean())

def _compute_group_mean(features, present_dict, names):
    stacked_feats = torch.stack([features[name] for name in names], dim=1)
    stacked_mask = torch.stack([present_dict[name] for name in names], dim=1).float()
    denom = stacked_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_feat = (stacked_feats * stacked_mask.unsqueeze(-1)).sum(dim=1) / denom
    valid_mask = stacked_mask.sum(dim=1) > 0
    return (mean_feat, valid_mask)

def compute_disentangle_regularization(aux, device):
    zero = torch.tensor(0.0, device=device)
    if aux is None:
        return {'structure': zero, 'orthogonality': zero, 'response_diversity': zero}
    struct_features = aux.get('struct_features')
    response_features = aux.get('response_features')
    present_dict = aux.get('present_dict')
    if not struct_features or not response_features or (not present_dict):
        return {'structure': zero, 'orthogonality': zero, 'response_diversity': zero}
    structure_losses = []
    for left, right in combinations(MRI_MODALITIES, 2):
        valid_mask = (present_dict[left] > 0.5) & (present_dict[right] > 0.5)
        _append_masked_cosine_distance(structure_losses, struct_features[left], struct_features[right], valid_mask, weight=1.0)
    structure_loss = torch.stack(structure_losses).mean() if structure_losses else zero
    orth_losses = []
    for name in MODALITY_ORDER:
        valid_mask = present_dict[name] > 0.5
        if valid_mask.any():
            cos_sim = F.cosine_similarity(struct_features[name][valid_mask], response_features[name][valid_mask], dim=-1)
            orth_losses.append(cos_sim.abs().mean())
    orth_loss = torch.stack(orth_losses).mean() if orth_losses else zero
    diversity_losses = []
    for left, right in combinations(MODALITY_ORDER, 2):
        valid_mask = (present_dict[left] > 0.5) & (present_dict[right] > 0.5)
        if valid_mask.any():
            cos_sim = F.cosine_similarity(response_features[left][valid_mask], response_features[right][valid_mask], dim=-1)
            diversity_losses.append(F.relu(cos_sim - 0.2).mean())
    diversity_loss = torch.stack(diversity_losses).mean() if diversity_losses else zero
    return {'structure': structure_loss, 'orthogonality': orth_loss, 'response_diversity': diversity_loss}

def compute_aux_distill_loss(teacher_aux, student_aux, device, sample_mask=None):
    zero = torch.tensor(0.0, device=device)
    if teacher_aux is None or student_aux is None:
        return (zero, zero)
    teacher_struct = teacher_aux.get('struct_features')
    student_struct = student_aux.get('struct_features')
    teacher_response = teacher_aux.get('response_features')
    student_response = student_aux.get('response_features')
    teacher_present = teacher_aux.get('present_dict')
    student_present = student_aux.get('present_dict')
    if not all([teacher_struct, student_struct, teacher_response, student_response, teacher_present, student_present]):
        return (zero, zero)
    struct_losses = []
    response_losses = []
    for name in MODALITY_ORDER:
        valid_mask = (teacher_present[name] > 0.5) & (student_present[name] > 0.5)
        if sample_mask is not None:
            valid_mask = valid_mask & sample_mask
        if valid_mask.any():
            struct_losses.append(compute_alignment_loss(teacher_struct[name][valid_mask], student_struct[name][valid_mask]))
            response_losses.append(compute_alignment_loss(teacher_response[name][valid_mask], student_response[name][valid_mask]))
    struct_loss = torch.stack(struct_losses).mean() if struct_losses else zero
    response_loss = torch.stack(response_losses).mean() if response_losses else zero
    return (struct_loss, response_loss)

def compute_pet_proxy_residual_kd_loss(teacher_aux, student_aux, real_ct_present, student_ct_present, device, sample_mask=None):
    zero = torch.tensor(0.0, device=device)
    if teacher_aux is None or student_aux is None:
        return (zero, zero)
    teacher_token = teacher_aux.get('pet_residual_token')
    teacher_weight = teacher_aux.get('pet_residual_weight')
    student_token = student_aux.get('pet_proxy_residual_token')
    student_weight = student_aux.get('pet_proxy_residual_weight')
    if teacher_token is None or teacher_weight is None or student_token is None or (student_weight is None):
        return (zero, zero)
    valid_mask = (real_ct_present['pet'] > 0.5) & (student_ct_present['pet'] < 0.5)
    student_proxy_mask = student_aux.get('pet_proxy_mask')
    if student_proxy_mask is not None:
        valid_mask = valid_mask & (student_proxy_mask.view(-1) > 0.5)
    if sample_mask is not None:
        valid_mask = valid_mask & sample_mask
    if not valid_mask.any():
        return (zero, zero)
    token_loss = compute_alignment_loss(teacher_token[valid_mask], student_token[valid_mask])
    weight_loss = F.mse_loss(student_weight[valid_mask], teacher_weight.detach()[valid_mask])
    return (token_loss, weight_loss)

def compute_pet_proxy_delta_kd_loss(teacher_aux, student_aux, real_ct_present, student_ct_present, device, sample_mask=None):
    zero = torch.tensor(0.0, device=device)
    if teacher_aux is None or student_aux is None:
        return zero
    teacher_response = teacher_aux.get('response_features')
    student_delta = student_aux.get('pet_proxy_delta_response')
    if teacher_response is None or student_delta is None:
        return zero
    if 'ct' not in teacher_response or 'pet' not in teacher_response:
        return zero
    valid_mask = (real_ct_present['ct'] > 0.5) & (real_ct_present['pet'] > 0.5) & (student_ct_present['pet'] < 0.5)
    if sample_mask is not None:
        valid_mask = valid_mask & sample_mask
    if not valid_mask.any():
        return zero
    teacher_delta = teacher_response['pet'] - teacher_response['ct']
    return compute_alignment_loss(teacher_delta[valid_mask], student_delta[valid_mask])

def compute_teacher_pet_effect_kd_loss(teacher_logits_full, teacher_logits_no_pet, student_aux, real_ct_present, student_ct_present, device, sample_mask=None, labels=None, class_balance=False, positive_weight=0.0, beneficial_only=False, benefit_min=0.02):
    zero = torch.tensor(0.0, device=device)
    if teacher_logits_full is None or teacher_logits_no_pet is None or student_aux is None:
        return zero
    student_effect_logit = student_aux.get('pet_effect_logit')
    if student_effect_logit is None:
        return zero
    valid_mask = (real_ct_present['ct'] > 0.5) & (real_ct_present['pet'] > 0.5) & (student_ct_present['pet'] < 0.5)
    student_proxy_mask = student_aux.get('pet_proxy_mask')
    if student_proxy_mask is not None:
        valid_mask = valid_mask & (student_proxy_mask.view(-1) > 0.5)
    if sample_mask is not None:
        valid_mask = valid_mask & sample_mask
    if not valid_mask.any():
        return zero
    teacher_effect_logit = teacher_logits_full.detach() - teacher_logits_no_pet.detach()
    if beneficial_only:
        if labels is None:
            raise ValueError('labels are required when beneficial_only=True')
        label_direction = labels.float().view(-1) * 2.0 - 1.0
        teacher_margin_gain = teacher_effect_logit.view(-1) * label_direction
        valid_mask = valid_mask & (teacher_margin_gain >= max(float(benefit_min), 0.0))
        if not valid_mask.any():
            return zero
    per_sample_loss = F.smooth_l1_loss(student_effect_logit[valid_mask].view(-1), teacher_effect_logit[valid_mask].view(-1), reduction='none')
    if labels is not None and float(positive_weight) > 0:
        valid_labels = labels.float().view(-1)[valid_mask]
        weights = torch.where(valid_labels > 0.5, torch.full_like(valid_labels, float(positive_weight)), torch.ones_like(valid_labels))
        return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-06)
    if not class_balance or labels is None:
        return per_sample_loss.mean()
    valid_labels = labels.float().view(-1)[valid_mask]
    positive = valid_labels > 0.5
    negative = ~positive
    if positive.any() and negative.any():
        return 0.5 * (per_sample_loss[positive].mean() + per_sample_loss[negative].mean())
    return per_sample_loss.mean()

def compute_teacher_pet_effect_sign_loss(teacher_logits_full, teacher_logits_no_pet, student_aux, labels, real_ct_present, student_ct_present, device, sample_mask=None, class_balance=False, positive_weight=0.0, beneficial_only=False, margin=0.02, min_target=0.02, temperature=0.02):
    zero = torch.tensor(0.0, device=device)
    if teacher_logits_full is None or teacher_logits_no_pet is None or student_aux is None:
        return zero
    student_effect_logit = student_aux.get('pet_effect_logit')
    if student_effect_logit is None:
        return zero
    valid_mask = (real_ct_present['ct'] > 0.5) & (real_ct_present['pet'] > 0.5) & (student_ct_present['pet'] < 0.5)
    student_proxy_mask = student_aux.get('pet_proxy_mask')
    if student_proxy_mask is not None:
        valid_mask = valid_mask & (student_proxy_mask.view(-1) > 0.5)
    if sample_mask is not None:
        valid_mask = valid_mask & sample_mask
    teacher_effect = (teacher_logits_full.detach() - teacher_logits_no_pet.detach()).view(-1)
    if beneficial_only:
        label_direction = labels.float().view(-1) * 2.0 - 1.0
        teacher_margin_gain = teacher_effect * label_direction
        valid_mask = valid_mask & (teacher_margin_gain >= max(float(min_target), 0.0))
    else:
        valid_mask = valid_mask & (teacher_effect.abs() >= max(float(min_target), 0.0))
    if not valid_mask.any():
        return zero
    target_direction = teacher_effect[valid_mask].sign()
    student_effect = student_effect_logit.view(-1)[valid_mask]
    temperature = max(float(temperature), 1e-06)
    per_sample_loss = F.softplus((float(margin) - target_direction * student_effect) / temperature) * temperature
    valid_labels = labels.float().view(-1)[valid_mask]
    if float(positive_weight) > 0:
        weights = torch.where(valid_labels > 0.5, torch.full_like(valid_labels, float(positive_weight)), torch.ones_like(valid_labels))
        return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-06)
    if not class_balance:
        return per_sample_loss.mean()
    positive = valid_labels > 0.5
    negative = ~positive
    if positive.any() and negative.any():
        return 0.5 * (per_sample_loss[positive].mean() + per_sample_loss[negative].mean())
    return per_sample_loss.mean()

def compute_teacher_pet_utility_gate_loss(teacher_logits_full, teacher_logits_no_pet, student_aux, labels, real_ct_present, student_ct_present, device, temperature=0.1, sample_mask=None):
    zero = torch.tensor(0.0, device=device)
    if teacher_logits_full is None or teacher_logits_no_pet is None or student_aux is None:
        return (zero, zero)
    student_gate = student_aux.get('pet_effect_weight')
    if student_gate is None:
        return (zero, zero)
    valid_mask = (real_ct_present['ct'] > 0.5) & (real_ct_present['pet'] > 0.5) & (student_ct_present['pet'] < 0.5)
    student_proxy_mask = student_aux.get('pet_proxy_mask')
    if student_proxy_mask is not None:
        valid_mask = valid_mask & (student_proxy_mask.view(-1) > 0.5)
    if sample_mask is not None:
        valid_mask = valid_mask & sample_mask
    if not valid_mask.any():
        return (zero, zero)
    temperature = max(float(temperature), 1e-06)
    with torch.no_grad():
        target_labels = labels.float().view(-1)
        full_loss = F.binary_cross_entropy_with_logits(teacher_logits_full.detach().view(-1), target_labels, reduction='none')
        no_pet_loss = F.binary_cross_entropy_with_logits(teacher_logits_no_pet.detach().view(-1), target_labels, reduction='none')
        utility = no_pet_loss - full_loss
        utility_target = torch.tanh(F.relu(utility) / temperature)
    gate = student_gate.view(-1)[valid_mask].clamp(0.0, 1.0)
    target = utility_target[valid_mask]
    return (F.smooth_l1_loss(gate, target), target.mean())

def compute_pet_residual_supervision_losses(student_aux, labels, device):
    zero = torch.tensor(0.0, device=device)
    if student_aux is None:
        return (zero, zero)
    base_logits = student_aux.get('base_logits')
    effect_logits = student_aux.get('pet_effect_logit')
    proxy_mask = student_aux.get('pet_proxy_mask')
    if base_logits is None or effect_logits is None:
        return (zero, zero)
    valid_mask = torch.ones_like(labels, dtype=torch.bool).view(-1)
    if proxy_mask is not None:
        valid_mask = valid_mask & (proxy_mask.view(-1) > 0.5)
    if not valid_mask.any():
        return (zero, zero)
    base = base_logits.detach().view(-1)[valid_mask]
    effect = effect_logits.view(-1)[valid_mask]
    target = labels.float().view(-1)[valid_mask] - torch.sigmoid(base)
    residual_loss = F.smooth_l1_loss(effect, target.detach())
    if effect.numel() < 3:
        return (residual_loss, zero)
    base_centered = base - base.mean()
    effect_centered = effect - effect.mean()
    denom = base_centered.norm() * effect_centered.norm()
    if denom.detach().item() <= 1e-08:
        return (residual_loss, zero)
    correlation = torch.dot(base_centered, effect_centered) / denom.clamp_min(1e-08)
    return (residual_loss, correlation.pow(2))

def compute_generated_response_kd_loss(teacher_aux, student_aux, real_mri_present, real_ct_present, student_mri_present, student_ct_present, device, cos_weight=0.5, t1c_weight=1.0, pet_weight=0.7):
    zero = torch.tensor(0.0, device=device)
    if teacher_aux is None or student_aux is None:
        return zero
    teacher_response = teacher_aux.get('response_features')
    student_response = student_aux.get('response_features')
    if not teacher_response or not student_response:
        return zero
    modality_specs = [('t1c', real_mri_present['t1c'], student_mri_present['t1c'], t1c_weight), ('pet', real_ct_present['pet'], student_ct_present['pet'], pet_weight)]
    losses = []
    weights = []
    for name, real_present, student_present, modality_weight in modality_specs:
        valid_mask = (real_present > 0.5) & (student_present < 0.5)
        if valid_mask.any():
            student_feat = student_response[name][valid_mask]
            teacher_feat = teacher_response[name][valid_mask].detach()
            mse_loss = F.mse_loss(student_feat, teacher_feat)
            cos_loss = 1.0 - F.cosine_similarity(student_feat, teacher_feat, dim=-1).mean()
            losses.append(modality_weight * (mse_loss + cos_weight * cos_loss))
            weights.append(torch.tensor(modality_weight, device=device))
    if not losses:
        return zero
    return torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1e-06)

def compute_meanflow_response_loss(teacher_aux, student_aux, real_mri_present, real_ct_present, student_mri_present, student_ct_present, device, cos_weight=0.2, endpoint_weight=0.5, max_time=0.8, t1c_weight=1.0, pet_weight=0.7):
    zero = torch.tensor(0.0, device=device)
    if teacher_aux is None or student_aux is None:
        return zero
    teacher_response = teacher_aux.get('response_features')
    meanflow_features = student_aux.get('meanflow_features')
    if not teacher_response or not meanflow_features:
        return zero
    modality_specs = [('t1c', real_mri_present['t1c'], student_mri_present['t1c'], t1c_weight), ('pet', real_ct_present['pet'], student_ct_present['pet'], pet_weight)]
    losses = []
    weights = []
    for name, real_present, student_present, modality_weight in modality_specs:
        flow_aux = meanflow_features.get(name)
        if not flow_aux:
            continue
        valid_mask = (real_present > 0.5) & (student_present < 0.5)
        if not valid_mask.any():
            continue
        required_keys = ['generator', 'target_struct', 'source_responses', 'source_masks', 'meta_feat', 'anchor_response']
        if any((key not in flow_aux for key in required_keys)):
            continue
        generator = flow_aux['generator']
        anchor_response = flow_aux['anchor_response'][valid_mask]
        target_response = teacher_response[name][valid_mask].detach()
        batch_size = target_response.size(0)
        t = torch.rand((batch_size, 1), device=device, dtype=target_response.dtype)
        t = t * max(0.0, min(float(max_time), 0.98))
        remaining = (1.0 - t).clamp_min(0.0001)
        flow_state = ((1.0 - t) * anchor_response.detach() + t * target_response).detach()
        _, sampled_flow_aux = generator(target_struct=flow_aux['target_struct'][valid_mask], source_responses=[src[valid_mask] for src in flow_aux['source_responses']], source_masks=[mask[valid_mask] for mask in flow_aux['source_masks']], meta_feat=flow_aux['meta_feat'][valid_mask], anchor_response=anchor_response, flow_state=flow_state, flow_time=t, target_name=flow_aux.get('target_name', name), return_aux=True)
        predicted_velocity = sampled_flow_aux['velocity']
        target_velocity = (target_response - flow_state) / remaining
        velocity_loss = F.smooth_l1_loss(predicted_velocity, target_velocity)
        cos_loss = 1.0 - F.cosine_similarity(predicted_velocity, target_velocity, dim=-1).mean()
        endpoint_pred = flow_state + remaining * predicted_velocity
        endpoint_loss = F.smooth_l1_loss(endpoint_pred, target_response)
        losses.append(modality_weight * (velocity_loss + endpoint_weight * endpoint_loss + cos_weight * cos_loss))
        weights.append(torch.tensor(modality_weight, device=device))
    if not losses:
        return zero
    return torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1e-06)

def compute_generation_quality_loss(teacher_aux, student_aux, real_mri_present, real_ct_present, student_mri_present, student_ct_present, device, temperature=1.0, t1c_weight=1.0, pet_weight=0.7):
    zero = torch.tensor(0.0, device=device)
    if teacher_aux is None or student_aux is None:
        return zero
    teacher_response = teacher_aux.get('response_features')
    meanflow_features = student_aux.get('meanflow_features')
    if not teacher_response or not meanflow_features:
        return zero
    modality_specs = [('t1c', real_mri_present['t1c'], student_mri_present['t1c'], t1c_weight), ('pet', real_ct_present['pet'], student_ct_present['pet'], pet_weight)]
    losses = []
    weights = []
    temperature = max(float(temperature), 1e-06)
    for name, real_present, student_present, modality_weight in modality_specs:
        flow_aux = meanflow_features.get(name)
        if not flow_aux or 'quality' not in flow_aux or 'generated_response' not in flow_aux:
            continue
        valid_mask = (real_present > 0.5) & (student_present < 0.5)
        if not valid_mask.any():
            continue
        generated_response = flow_aux['generated_response'][valid_mask]
        teacher_feat = teacher_response[name][valid_mask].detach()
        quality_pred = flow_aux['quality'][valid_mask].view(-1).clamp(0.0001, 1.0 - 0.0001)
        error = torch.norm(generated_response - teacher_feat, dim=-1) / teacher_feat.size(-1) ** 0.5
        quality_target = torch.exp((-error / temperature).detach()).clamp(0.0, 1.0)
        losses.append(modality_weight * F.binary_cross_entropy(quality_pred, quality_target))
        weights.append(torch.tensor(modality_weight, device=device))
    if not losses:
        return zero
    return torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1e-06)

def get_warmup_weight(base_weight, epoch, warmup_epochs):
    if base_weight <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return base_weight
    scale = min(1.0, float(epoch + 1) / float(warmup_epochs))
    return base_weight * scale

def compute_pet_gain_weights(teacher_logits_full, teacher_logits_no_pet, valid_mask, alpha):
    gain = (teacher_logits_full.detach() - teacher_logits_no_pet.detach()).abs().view(-1)
    weights = torch.ones_like(gain)
    if valid_mask.any():
        gain_scale = gain[valid_mask].mean().clamp_min(1e-06)
        normalized_gain = (gain / gain_scale).clamp(0.0, 5.0)
        weights = 1.0 + alpha * normalized_gain
    return weights

def compute_pet_gain_weighted_logit_kd_loss(student_logits, teacher_logits_full, teacher_logits_no_pet, valid_mask, alpha, device):
    if not valid_mask.any():
        return (torch.tensor(0.0, device=device), torch.ones(student_logits.size(0), device=device))
    sample_weights = compute_pet_gain_weights(teacher_logits_full, teacher_logits_no_pet, valid_mask, alpha)
    per_sample_loss = (student_logits.view(-1) - teacher_logits_full.detach().view(-1)).pow(2)
    weighted_loss = (sample_weights[valid_mask] * per_sample_loss[valid_mask]).sum() / sample_weights[valid_mask].sum().clamp_min(1e-06)
    return (weighted_loss, sample_weights)

def compute_pet_gain_ranking_kd_loss(student_logits, teacher_logits_full, valid_mask, sample_weights, margin, device):
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)
    teacher_scores = teacher_logits_full.detach().view(-1)[valid_mask]
    student_scores = student_logits.view(-1)[valid_mask]
    ranking_weights = sample_weights.view(-1)[valid_mask]
    if teacher_scores.numel() < 2:
        return torch.tensor(0.0, device=device)
    teacher_diff = teacher_scores.unsqueeze(1) - teacher_scores.unsqueeze(0)
    student_diff = student_scores.unsqueeze(1) - student_scores.unsqueeze(0)
    pair_mask = torch.triu(torch.ones_like(teacher_diff, dtype=torch.bool), diagonal=1)
    informative_mask = pair_mask & (teacher_diff.abs() > 1e-06)
    if not informative_mask.any():
        return torch.tensor(0.0, device=device)
    target_sign = torch.sign(teacher_diff[informative_mask])
    pair_losses = F.relu(margin - target_sign * student_diff[informative_mask])
    pair_weight_matrix = 0.5 * (ranking_weights.unsqueeze(1) + ranking_weights.unsqueeze(0))
    pair_weights = pair_weight_matrix[informative_mask]
    return (pair_weights * pair_losses).sum() / pair_weights.sum().clamp_min(1e-06)

def train_teacher_one_epoch(model, dataloader, criterion, optimizer, device, input_mode, disable_meta, struct_consistency_weight, orthogonality_weight, response_diversity_weight):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc='Teacher Training', leave=False)
    for batch in pbar:
        mri_imgs, ct_pet_imgs, mri_present, ct_present, vols, regions, labels = extract_batch(batch, device)
        mri_imgs, mri_present, ct_pet_imgs, ct_present = apply_input_mode(mri_imgs, mri_present, ct_pet_imgs, ct_present, input_mode)
        vols, regions = apply_meta_ablation(vols, regions, disable_meta)
        optimizer.zero_grad()
        logits, _, _ = model(mri_imgs, mri_present, ct_pet_imgs, ct_present, vols, regions)
        aux_losses = compute_disentangle_regularization(get_model_aux(model), device)
        loss_cls = criterion(logits, labels)
        loss = loss_cls + struct_consistency_weight * aux_losses['structure'] + orthogonality_weight * aux_losses['orthogonality'] + response_diversity_weight * aux_losses['response_diversity']
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'L_cls': f'{loss_cls.item():.3f}', 'L_str': f"{aux_losses['structure'].item():.3f}", 'L_orth': f"{aux_losses['orthogonality'].item():.3f}", 'L_div': f"{aux_losses['response_diversity'].item():.3f}"})
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return running_loss / len(dataloader)

def train_student_one_epoch(teacher_model, student_model, dataloader, criterion, optimizer, device, kd_logits_weight, kd_feature_weight, disable_meta, struct_consistency_weight, orthogonality_weight, response_diversity_weight, student_struct_kd_weight, student_response_kd_weight, student_base_cls_weight, student_pet_proxy_kd_weight, student_pet_proxy_weight_kd_weight, student_pet_delta_kd_weight, student_pet_effect_kd_weight, student_pet_effect_class_balance, student_pet_effect_positive_weight, student_pet_effect_beneficial_only, student_pet_effect_benefit_min, student_pet_effect_sign_weight, student_pet_effect_sign_margin, student_pet_effect_sign_min_target, student_pet_effect_sign_temperature, student_pet_utility_weight, student_pet_utility_temperature, student_pet_residual_weight, student_pet_decorrelation_weight, student_pet_gain_logit_kd_weight, student_pet_ranking_kd_weight, student_pet_gain_alpha, student_pet_ranking_margin, generated_response_kd_weight, generated_response_cos_weight, generated_response_t1c_weight, generated_response_pet_weight, generation_quality_weight, generation_quality_temperature, meanflow_response_weight, meanflow_response_cos_weight, meanflow_response_endpoint_weight, meanflow_response_max_time, meanflow_response_t1c_weight, meanflow_response_pet_weight, hide_prob_t1c, hide_prob_pet, input_mode, teacher_mri_volume_mode='current_log', student_supervised_ranking_weight=0.0, student_supervised_ranking_margin=0.1):
    teacher_model.eval()
    student_model.train()
    base_student_model = student_model.module if isinstance(student_model, DDP) else student_model
    if getattr(base_student_model, 'cpea_stage2_frozen', False):
        base_student_model.eval()
        base_student_model.pet_response_generator.train()
        base_student_model.pet_effect_adapter.train()
        base_student_model.feature_fusers['pet'].train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc='Student Distillation', leave=False)
    for batch in pbar:
        mri_imgs, ct_pet_imgs, mri_present, ct_present, vols, regions, labels = extract_batch(batch, device)
        vols, regions = apply_meta_ablation(vols, regions, disable_meta)
        student_mri_present, student_ct_present = make_student_present_dict(mri_present, ct_present, hide_prob_t1c=hide_prob_t1c, hide_prob_pet=hide_prob_pet)
        student_mri_present, student_ct_present = apply_student_input_mode(student_mri_present, student_ct_present, input_mode)
        student_mri_present, student_ct_present = apply_source_family_dropout(student_mri_present, student_ct_present)
        student_vols = mask_unobserved_mri_volume(vols, student_mri_present)
        student_mri_imgs, student_ct_pet_imgs = mask_images_by_present(mri_imgs, ct_pet_imgs, student_mri_present, student_ct_present)
        privileged_mask = get_full_modality_mask(mri_present, ct_present)
        teacher_vols = prepare_teacher_mri_volumes(vols, teacher_mri_volume_mode)
        with torch.no_grad():
            t_logits, t_out_sequence, t_features = teacher_model(mri_imgs, mri_present, ct_pet_imgs, ct_present, teacher_vols, regions)
            teacher_aux = get_model_aux(teacher_model)
            t_logits_no_pet = None
            teacher_aux_no_pet = None
            if student_pet_effect_kd_weight > 0 or student_pet_effect_sign_weight > 0 or student_pet_utility_weight > 0 or (student_pet_gain_logit_kd_weight > 0) or (student_pet_ranking_kd_weight > 0):
                ct_present_no_pet = dict(ct_present)
                ct_present_no_pet['pet'] = torch.zeros_like(ct_present['pet'])
                ct_pet_imgs_no_pet = ct_pet_imgs.clone() if ct_pet_imgs is not None else None
                if ct_pet_imgs_no_pet is not None and ct_pet_imgs_no_pet.size(1) > 1:
                    ct_pet_imgs_no_pet[:, 1:2, ...] = 0
                t_logits_no_pet, _, _ = teacher_model(mri_imgs, mri_present, ct_pet_imgs_no_pet, ct_present_no_pet, teacher_vols, regions)
                teacher_aux_no_pet = get_model_aux(teacher_model)
        optimizer.zero_grad()
        s_logits, s_out_sequence, s_features = student_model(student_mri_imgs, student_mri_present, student_ct_pet_imgs, student_ct_present, student_vols, regions)
        student_aux = get_model_aux(student_model)
        loss_cls = criterion(s_logits, labels)
        loss_risk_ranking = supervised_risk_ranking_loss(s_logits, labels, margin=student_supervised_ranking_margin) if student_supervised_ranking_weight > 0 else s_logits.sum() * 0.0
        base_logits = student_aux.get('base_logits') if student_aux is not None else None
        loss_base_cls = criterion(base_logits, labels) if base_logits is not None else torch.tensor(0.0, device=device)
        loss_kd_logits = masked_mse_loss(s_logits, t_logits, privileged_mask, device)
        loss_kd_feat = torch.tensor(0.0, device=device)
        valid_targets = 0
        fake_t1c_indices = ((mri_present['t1c'] > 0.5) & (student_mri_present['t1c'] < 0.5) & privileged_mask).nonzero(as_tuple=True)[0]
        if len(fake_t1c_indices) > 0:
            teacher_t1c_token = t_out_sequence[fake_t1c_indices, 1, :]
            student_t1c_token = s_out_sequence[fake_t1c_indices, 1, :]
            loss_kd_feat = loss_kd_feat + compute_alignment_loss(teacher_t1c_token, student_t1c_token)
            valid_targets += 1
        fake_pet_indices = ((ct_present['pet'] > 0.5) & (student_ct_present['pet'] < 0.5) & privileged_mask).nonzero(as_tuple=True)[0]
        if len(fake_pet_indices) > 0:
            teacher_pet_token = t_out_sequence[fake_pet_indices, 4, :]
            student_pet_proxy = student_aux.get('pet_proxy_token') if student_aux is not None else None
            student_pet_token = student_pet_proxy[fake_pet_indices] if student_pet_proxy is not None else s_out_sequence[fake_pet_indices, 4, :]
            loss_kd_feat = loss_kd_feat + compute_alignment_loss(teacher_pet_token, student_pet_token)
            valid_targets += 1
        if valid_targets > 0:
            loss_kd_feat = loss_kd_feat / valid_targets
        student_aux_losses = compute_disentangle_regularization(student_aux, device)
        loss_struct_kd, loss_response_kd = compute_aux_distill_loss(teacher_aux, student_aux, device, sample_mask=privileged_mask)
        loss_pet_proxy_kd, loss_pet_proxy_weight_kd = compute_pet_proxy_residual_kd_loss(teacher_aux=teacher_aux, student_aux=student_aux, real_ct_present=ct_present, student_ct_present=student_ct_present, device=device, sample_mask=privileged_mask)
        loss_pet_delta_kd = compute_pet_proxy_delta_kd_loss(teacher_aux=teacher_aux, student_aux=student_aux, real_ct_present=ct_present, student_ct_present=student_ct_present, device=device, sample_mask=privileged_mask)
        loss_pet_effect_kd = compute_teacher_pet_effect_kd_loss(teacher_logits_full=t_logits, teacher_logits_no_pet=t_logits_no_pet, student_aux=student_aux, real_ct_present=ct_present, student_ct_present=student_ct_present, device=device, sample_mask=privileged_mask, labels=labels, class_balance=student_pet_effect_class_balance, positive_weight=student_pet_effect_positive_weight, beneficial_only=student_pet_effect_beneficial_only, benefit_min=student_pet_effect_benefit_min)
        loss_pet_effect_sign = compute_teacher_pet_effect_sign_loss(teacher_logits_full=t_logits, teacher_logits_no_pet=t_logits_no_pet, student_aux=student_aux, labels=labels, real_ct_present=ct_present, student_ct_present=student_ct_present, device=device, sample_mask=privileged_mask, class_balance=student_pet_effect_class_balance, positive_weight=student_pet_effect_positive_weight, beneficial_only=student_pet_effect_beneficial_only, margin=student_pet_effect_sign_margin, min_target=student_pet_effect_sign_min_target, temperature=student_pet_effect_sign_temperature)
        loss_pet_utility, pet_utility_target_mean = compute_teacher_pet_utility_gate_loss(teacher_logits_full=t_logits, teacher_logits_no_pet=t_logits_no_pet, student_aux=student_aux, labels=labels, real_ct_present=ct_present, student_ct_present=student_ct_present, device=device, temperature=student_pet_utility_temperature, sample_mask=privileged_mask)
        loss_pet_residual, loss_pet_decorrelation = compute_pet_residual_supervision_losses(student_aux=student_aux, labels=labels, device=device)
        pet_gain_valid_mask = (ct_present['pet'] > 0.5) & (student_ct_present['pet'] < 0.5) & privileged_mask
        loss_pet_gain_logit_kd, pet_gain_sample_weights = compute_pet_gain_weighted_logit_kd_loss(student_logits=s_logits, teacher_logits_full=t_logits, teacher_logits_no_pet=t_logits_no_pet if t_logits_no_pet is not None else t_logits, valid_mask=pet_gain_valid_mask, alpha=student_pet_gain_alpha, device=device)
        loss_pet_ranking_kd = compute_pet_gain_ranking_kd_loss(student_logits=s_logits, teacher_logits_full=t_logits, valid_mask=pet_gain_valid_mask, sample_weights=pet_gain_sample_weights, margin=student_pet_ranking_margin, device=device)
        loss_generated_response_kd = compute_generated_response_kd_loss(teacher_aux=teacher_aux, student_aux=student_aux, real_mri_present=mri_present, real_ct_present=ct_present, student_mri_present=student_mri_present, student_ct_present=student_ct_present, device=device, cos_weight=generated_response_cos_weight, t1c_weight=generated_response_t1c_weight, pet_weight=generated_response_pet_weight)
        loss_generation_quality = compute_generation_quality_loss(teacher_aux=teacher_aux, student_aux=student_aux, real_mri_present=mri_present, real_ct_present=ct_present, student_mri_present=student_mri_present, student_ct_present=student_ct_present, device=device, temperature=generation_quality_temperature, t1c_weight=generated_response_t1c_weight, pet_weight=generated_response_pet_weight)
        total_loss = loss_cls + student_supervised_ranking_weight * loss_risk_ranking + student_base_cls_weight * loss_base_cls + kd_logits_weight * loss_kd_logits + kd_feature_weight * loss_kd_feat + struct_consistency_weight * student_aux_losses['structure'] + orthogonality_weight * student_aux_losses['orthogonality'] + response_diversity_weight * student_aux_losses['response_diversity'] + student_struct_kd_weight * loss_struct_kd + student_response_kd_weight * loss_response_kd + student_pet_proxy_kd_weight * loss_pet_proxy_kd + student_pet_proxy_weight_kd_weight * loss_pet_proxy_weight_kd + student_pet_delta_kd_weight * loss_pet_delta_kd + student_pet_effect_kd_weight * loss_pet_effect_kd + student_pet_effect_sign_weight * loss_pet_effect_sign + student_pet_utility_weight * loss_pet_utility + student_pet_residual_weight * loss_pet_residual + student_pet_decorrelation_weight * loss_pet_decorrelation + student_pet_gain_logit_kd_weight * loss_pet_gain_logit_kd + student_pet_ranking_kd_weight * loss_pet_ranking_kd + generated_response_kd_weight * loss_generated_response_kd + generation_quality_weight * loss_generation_quality
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
        pbar.set_postfix({'L_cls': f'{loss_cls.item():.3f}', 'L_base': f'{loss_base_cls.item():.3f}', 'L_log': f'{loss_kd_logits.item():.3f}', 'L_feat': f'{loss_kd_feat.item():.3f}' if valid_targets > 0 else '0.000', 'L_str': f"{student_aux_losses['structure'].item():.3f}", 'L_orth': f"{student_aux_losses['orthogonality'].item():.3f}", 'L_div': f"{student_aux_losses['response_diversity'].item():.3f}", 'L_ppx': f'{loss_pet_proxy_kd.item():.3f}', 'L_pdel': f'{loss_pet_delta_kd.item():.3f}', 'L_peff': f'{loss_pet_effect_kd.item():.3f}', 'L_psign': f'{loss_pet_effect_sign.item():.3f}', 'L_putil': f'{loss_pet_utility.item():.3f}', 'U_pet': f'{pet_utility_target_mean.item():.2f}', 'L_pres': f'{loss_pet_residual.item():.3f}', 'L_pdec': f'{loss_pet_decorrelation.item():.3f}', 'L_pgain': f'{loss_pet_gain_logit_kd.item():.3f}', 'L_prank': f'{loss_pet_ranking_kd.item():.3f}', 'L_rank': f'{loss_risk_ranking.item():.3f}', 'L_qual': f'{loss_generation_quality.item():.3f}', 'Priv': f'{privileged_mask.float().mean().item():.2f}'})
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return running_loss / len(dataloader)

def mask_unobserved_mri_volume(mri_volume, student_mri_present):
    observed = (student_mri_present['t1'].view(-1) > 0.5) | (student_mri_present['t2'].view(-1) > 0.5)
    value = mri_volume.reshape(-1, 1)
    assert len(value) == len(observed)
    return torch.where(observed[:, None], value, torch.zeros_like(value)).contiguous()
