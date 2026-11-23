import torch
import torch.nn as nn
import torch.nn.functional as F
from robustness import convert_response_norm, student_mri_style

def _instance_normalize_3d(x, eps=1e-05):
    dims = tuple(range(2, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = (x - mean).pow(2).mean(dim=dims, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)

def _gradient_magnitude_3d(x, eps=1e-08):
    dz = F.pad(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))
    dy = F.pad(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], (0, 0, 0, 1, 0, 0))
    dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
    return torch.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + eps)

def make_structure_guided_input(x, gradient_weight=0.5):
    normed = _instance_normalize_3d(x)
    gradient = _instance_normalize_3d(_gradient_magnitude_3d(normed))
    mixed = (1.0 - gradient_weight) * normed + gradient_weight * gradient
    return _instance_normalize_3d(mixed)

class structExtractor(nn.Module):

    def __init__(self, in_channels=1, feature_dim=128, dim=64, fs=3):
        super().__init__()
        self.block1 = nn.ModuleList()
        for _ in range(fs):
            self.block1.append(nn.Sequential(nn.Conv3d(in_channels, dim, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm3d(dim), nn.LeakyReLU(0.2, inplace=True)))
        self.block2 = nn.Sequential(nn.Conv3d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm3d(dim * 2), nn.LeakyReLU(0.2, inplace=True))
        self.fs = fs

    def forward(self, x):
        xl = []
        for i in range(self.fs):
            xl.append(self.block1[i](x[:, i:i + 1, ...]))
        x = torch.cat(xl, dim=1)
        b, total_c, d, h, w = x.shape
        c = total_c // self.fs
        x = x.view(b, self.fs, c, d, h, w).view(b * self.fs, c, d, h, w)
        x = self.block2(x)
        c_out = x.shape[1]
        d_new, h_new, w_new = x.shape[2:]
        x = x.view(b, self.fs, c_out, d_new, h_new, w_new).view(b, self.fs * c_out, d_new, h_new, w_new)
        return x

class _GlobalPool(nn.Module):

    def __init__(self):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

class FeatProjector(nn.Module):

    def __init__(self, in_channels=1, feature_dim=128, dim=64, fs=3):
        super().__init__()
        self.fcs = nn.ModuleList()
        pooled = _GlobalPool()
        for _ in range(fs):
            self.fcs.append(nn.Sequential(nn.Conv3d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm3d(dim * 4), nn.LeakyReLU(0.2, inplace=True), pooled, nn.Linear(dim * 4, feature_dim), nn.LayerNorm(feature_dim)))
        self.fs = fs
        self.dim = dim * 2

    def forward(self, x):
        xl = []
        for i in range(self.fs):
            tx = x[:, i * self.dim:(i + 1) * self.dim, :, :, :]
            xl.append(self.fcs[i](tx))
        return torch.cat(xl, dim=1)

class reactExtractor_base(nn.Module):

    def __init__(self, in_channels=1, feature_dim=128, dim=64, fs=3):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv3d(in_channels, dim, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm3d(dim), nn.LeakyReLU(0.2, inplace=True))
        self.block2 = nn.Sequential(nn.Conv3d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm3d(dim * 2), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class reactExtractor(nn.Module):

    def __init__(self, in_channels=1, feature_dim=128, dim=64, fs=3):
        super().__init__()
        self.block1 = nn.ModuleList()
        for _ in range(fs):
            self.block1.append(reactExtractor_base(in_channels, feature_dim, dim))
        self.fs = fs

    def forward(self, x):
        xl = []
        for i in range(self.fs):
            xl.append(self.block1[i](x[:, i:i + 1, :, :, :]))
        return torch.cat(xl, dim=1)

class FamilyMapFusion(nn.Module):

    def __init__(self, channels, out_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False), nn.BatchNorm3d(channels), nn.Sigmoid())
        hidden_channels = max(channels // 2, 64)
        self.refine = nn.Sequential(nn.Conv3d(channels, hidden_channels, kernel_size=1, bias=False), nn.BatchNorm3d(hidden_channels), nn.GELU(), nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(hidden_channels), nn.GELU())
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(nn.Linear(hidden_channels, out_dim), nn.LayerNorm(out_dim))

    def forward(self, struct_map, response_map):
        gate = self.gate(torch.cat([struct_map, response_map], dim=1))
        fused_map = gate * struct_map + (1.0 - gate) * response_map
        refined_map = self.refine(fused_map)
        pooled = self.pool(refined_map).flatten(1)
        return (self.proj(pooled), refined_map)

class PETResidualBranch(nn.Module):

    def __init__(self, token_dim, map_channels, token_fusion_dim, map_fusion_dim, meta_dim, hidden_dim=256):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.map_proj = nn.Sequential(nn.Linear(map_channels * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.feature_proj = nn.Sequential(nn.Linear(token_dim * 2 + hidden_dim + meta_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.residual_proj = nn.Sequential(nn.Linear(hidden_dim, token_fusion_dim), nn.GELU(), nn.LayerNorm(token_fusion_dim))
        self.weight_head = nn.Sequential(nn.Linear(hidden_dim + token_fusion_dim + map_fusion_dim + meta_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())

    def forward(self, pet_struct_token, pet_response_token, pet_response_map, token_repr, map_repr, meta_raw_feat, pet_present):
        avg_desc = self.avg_pool(pet_response_map).flatten(1)
        max_desc = self.max_pool(pet_response_map).flatten(1)
        map_desc = self.map_proj(torch.cat([avg_desc, max_desc], dim=1))
        residual_feature = self.feature_proj(torch.cat([pet_struct_token, pet_response_token, map_desc, meta_raw_feat], dim=1))
        residual_update = self.residual_proj(residual_feature)
        raw_weight = self.weight_head(torch.cat([residual_feature, token_repr, map_repr, meta_raw_feat], dim=1))
        pet_weight = raw_weight * pet_present.view(-1, 1)
        gated_residual = residual_update * pet_weight
        return (gated_residual, residual_feature, pet_weight, map_desc)

class AttentionPool(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, x):
        attn = torch.softmax(self.score(x), dim=1)
        return (attn * x).sum(dim=1)

class QualityPresenceTokenPool(nn.Module):

    def __init__(self, d_model, hidden_dim=64, source_type_dim=8):
        super().__init__()
        self.source_type_embedding = nn.Embedding(3, source_type_dim)
        self.score = nn.Sequential(nn.Linear(d_model * 2 + source_type_dim + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, modality_tokens, meta_token, present_mask, source_type_ids=None, quality_scores=None):
        if meta_token.dim() == 3:
            meta_token = meta_token.squeeze(1)
        b, num_modalities, _ = modality_tokens.shape
        present_mask = present_mask.float()
        if source_type_ids is None:
            source_type_ids = torch.where(present_mask > 0.5, torch.ones_like(present_mask, dtype=torch.long), torch.zeros_like(present_mask, dtype=torch.long))
        if quality_scores is None:
            quality_scores = present_mask
        source_type_emb = self.source_type_embedding(source_type_ids.long().clamp(0, 2))
        meta_expand = meta_token.unsqueeze(1).expand(-1, num_modalities, -1)
        score_input = torch.cat([modality_tokens, meta_expand, source_type_emb, present_mask.unsqueeze(-1), quality_scores.unsqueeze(-1)], dim=-1)
        logits = self.score(score_input).squeeze(-1)
        logits = logits.masked_fill(present_mask <= 0.5, -10000.0)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(modality_tokens * weights.unsqueeze(-1), dim=1)
        return (pooled, weights)

def build_source_type_tensor(source_type_dict, present_dict, modality_names, batch_size, device):
    values = []
    for name in modality_names:
        present = present_dict[name].to(device).view(batch_size)
        if source_type_dict is not None and name in source_type_dict:
            source_type = source_type_dict[name].to(device).long().view(batch_size)
        else:
            source_type = torch.where(present > 0.5, torch.ones_like(present, dtype=torch.long), torch.zeros_like(present, dtype=torch.long))
        values.append(source_type)
    return torch.stack(values, dim=1)

def build_quality_tensor(quality_dict, present_dict, modality_names, batch_size, device):
    values = []
    for name in modality_names:
        present = present_dict[name].to(device).float().view(batch_size)
        if quality_dict is not None and name in quality_dict:
            quality = quality_dict[name].to(device).float().view(batch_size)
        else:
            quality = present
        values.append(quality.clamp(0.0, 1.0))
    return torch.stack(values, dim=1)

class DualLevelTokenFusion(nn.Module):

    def __init__(self, cnn_feat_dim=128, meta_feat_dim=16, d_model=256, num_heads=4, num_layers=2, feature_dim=None):
        super().__init__()
        if feature_dim is not None:
            if cnn_feat_dim != 128 and cnn_feat_dim != feature_dim:
                raise ValueError('cnn_feat_dim and feature_dim must match when both are provided.')
            cnn_feat_dim = feature_dim
        self.feature_dim = cnn_feat_dim
        self.d_model = d_model
        self.proj_layers = nn.ModuleDict({'t1': nn.Linear(cnn_feat_dim, d_model), 't1c': nn.Linear(cnn_feat_dim, d_model), 't2': nn.Linear(cnn_feat_dim, d_model), 'ct': nn.Linear(cnn_feat_dim, d_model), 'pet': nn.Linear(cnn_feat_dim, d_model), 'meta': nn.Linear(meta_feat_dim, d_model)})
        self.modality_embs = nn.Parameter(torch.randn(6, 1, d_model) * 0.02)
        self.mask_tokens = nn.ParameterDict({'t1c': nn.Parameter(torch.randn(1, 1, d_model) * 0.02), 'pet': nn.Parameter(torch.randn(1, 1, d_model) * 0.02)})
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = AttentionPool(d_model)
        self.quality_pool = QualityPresenceTokenPool(d_model)
        self.output_fuser = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.latest_fusion_weights = None

    def forward(self, features_dict, present_dict, meta_raw_feat, source_type_dict=None, quality_dict=None):
        b = meta_raw_feat.size(0)
        modality_names = ['t1', 't1c', 't2', 'ct', 'pet']
        seqs = []
        present_mask = []
        for idx, name in enumerate(modality_names):
            token = self.proj_layers[name](features_dict[name]).unsqueeze(1)
            present = present_dict[name].view(b)
            if name in self.mask_tokens:
                mask = present.view(b, 1, 1).expand(-1, -1, self.d_model)
                token = torch.where(mask > 0.5, token, self.mask_tokens[name].expand(b, -1, -1))
            seqs.append(token + self.modality_embs[idx])
            present_mask.append(present)
        meta_token = self.proj_layers['meta'](meta_raw_feat).unsqueeze(1) + self.modality_embs[5]
        sequence = torch.cat(seqs + [meta_token], dim=1)
        out_sequence = self.transformer(sequence)
        pooled = self.attention_pool(out_sequence)
        present_mask = torch.stack(present_mask, dim=1)
        source_type_ids = build_source_type_tensor(source_type_dict, present_dict, modality_names, b, meta_raw_feat.device)
        quality_scores = build_quality_tensor(quality_dict, present_dict, modality_names, b, meta_raw_feat.device)
        quality_pooled, fusion_weights = self.quality_pool(out_sequence[:, :5, :], out_sequence[:, 5:6, :], present_mask, source_type_ids=source_type_ids, quality_scores=quality_scores)
        self.latest_fusion_weights = fusion_weights.detach()
        token_repr = self.output_fuser(torch.cat([pooled, quality_pooled], dim=1))
        return (token_repr, out_sequence)

class MultiModalLymphNodeNet(nn.Module):

    def __init__(self, num_regions=15, feature_dim=128, mri_response_norm='instance', mri_style_augmentation=True):
        super().__init__()
        inputdim = 96
        region_emb_dim = 8
        self.feature_dim = feature_dim
        self.fd = feature_dim
        self.structure_gradient_weight = 0.5
        self.ct_family_channels = inputdim * 2
        self.region_emb = nn.Embedding(num_regions, region_emb_dim)
        self.strct_extractor_mri = structExtractor(in_channels=1, feature_dim=feature_dim, dim=inputdim, fs=3)
        self.strct_extractor_ct = structExtractor(in_channels=1, feature_dim=feature_dim, dim=inputdim, fs=2)
        self.react_extractor_mri = reactExtractor(in_channels=1, feature_dim=feature_dim, dim=inputdim, fs=3)
        if mri_response_norm == 'instance':
            assert convert_response_norm(self.react_extractor_mri) == 6
        elif mri_response_norm != 'batch':
            raise ValueError('Unknown MRI response normalisation')
        self.mri_style_augmentation = bool(mri_style_augmentation)
        self.react_extractor_ct = reactExtractor(in_channels=1, feature_dim=feature_dim, dim=inputdim, fs=2)
        self.mp = FeatProjector(1, feature_dim, dim=inputdim, fs=3)
        self.cp = FeatProjector(1, feature_dim, dim=inputdim, fs=2)
        self.mri_map_fusion = FamilyMapFusion(channels=inputdim * 2 * 3, out_dim=feature_dim)
        self.ct_map_fusion = FamilyMapFusion(channels=inputdim * 2 * 2, out_dim=feature_dim)
        self.map_joint_fuser = nn.Sequential(nn.Linear(feature_dim * 2, 256), nn.GELU(), nn.LayerNorm(256))
        meta_input_dim = 1 + region_emb_dim
        meta_feature_dim = 16
        self.meta_encoder = nn.Sequential(nn.Linear(meta_input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=False), nn.Linear(32, meta_feature_dim), nn.ReLU(inplace=False))
        self.transformer_model = DualLevelTokenFusion(cnn_feat_dim=feature_dim, meta_feat_dim=meta_feature_dim, d_model=256, num_heads=4, num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(256 + 256 + meta_feature_dim, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))
        self.pet_residual_branch = PETResidualBranch(token_dim=feature_dim, map_channels=self.ct_family_channels, token_fusion_dim=256, map_fusion_dim=256, meta_dim=meta_feature_dim, hidden_dim=256)
        self.enable_aux_outputs = False
        self.latest_aux = None

    def _split_projected_features(self, projected_feat, names):
        output = {}
        for i, name in enumerate(names):
            output[name] = projected_feat[:, i * self.fd:(i + 1) * self.fd]
        return output

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        optional_prefixes = (f'{prefix}pet_residual_branch.',)
        missing_keys[:] = [key for key in missing_keys if not key.startswith(optional_prefixes)]

    def forward(self, mri_imgs, mri_present, ct_pet_imgs, ct_present, vols, regions):
        if self.training and self.mri_style_augmentation:
            mri_imgs = student_mri_style(mri_imgs)
        b = mri_imgs.size(0)
        present_dict = {'t1': mri_present['t1'].view(b), 't1c': mri_present['t1c'].view(b), 't2': mri_present['t2'].view(b), 'ct': ct_present['ct'].view(b), 'pet': ct_present['pet'].view(b)}
        mriimages = torch.cat((mri_imgs[:, 0:1, ...], mri_imgs[:, 1:2, ...], mri_imgs[:, 2:3, ...]), dim=1)
        ctimages = torch.cat((ct_pet_imgs[:, 0:1, ...], ct_pet_imgs[:, 1:2, ...]), dim=1)
        mri_struct_images = make_structure_guided_input(mriimages, self.structure_gradient_weight)
        ct_struct_images = make_structure_guided_input(ctimages, self.structure_gradient_weight)
        mrisf = self.strct_extractor_mri(mri_struct_images)
        ctsf = self.strct_extractor_ct(ct_struct_images)
        mrirf = self.react_extractor_mri(mriimages)
        ctrf = self.react_extractor_ct(ctimages)
        mf_struct = self.mp(mrisf)
        cf_struct = self.cp(ctsf)
        mf_response = self.mp(mrirf)
        cf_response = self.cp(ctrf)
        mf = self.mp(mrisf + mrirf)
        cf = self.cp(ctsf + ctrf)
        features_dict = {'t1': mf[:, 0 * self.fd:1 * self.fd], 't1c': mf[:, 1 * self.fd:2 * self.fd], 't2': mf[:, 2 * self.fd:3 * self.fd], 'ct': cf[:, 0 * self.fd:1 * self.fd], 'pet': cf[:, 1 * self.fd:2 * self.fd]}
        mri_map_repr, _ = self.mri_map_fusion(mrisf, mrirf)
        ct_map_repr, _ = self.ct_map_fusion(ctsf, ctrf)
        map_repr = self.map_joint_fuser(torch.cat([mri_map_repr, ct_map_repr], dim=1))
        reg_emb = self.region_emb(regions.long().view(b))
        meta_input = torch.cat([vols.view(b, 1), reg_emb], dim=1)
        meta_raw_feat = self.meta_encoder(meta_input)
        token_repr, out_sequence = self.transformer_model(features_dict, present_dict, meta_raw_feat)
        pet_struct_token = cf_struct[:, 1 * self.fd:2 * self.fd]
        pet_response_token = cf_response[:, 1 * self.fd:2 * self.fd]
        pet_response_map = ctrf[:, self.ct_family_channels:self.ct_family_channels * 2, :, :, :]
        pet_residual_token, pet_residual_feature, pet_residual_weight, pet_map_descriptor = self.pet_residual_branch(pet_struct_token, pet_response_token, pet_response_map, token_repr, map_repr, meta_raw_feat, present_dict['pet'])
        token_repr = token_repr + pet_residual_token
        base_context = torch.cat([map_repr, token_repr, meta_raw_feat], dim=1)
        base_logits = self.classifier(base_context)
        logits = base_logits
        if self.enable_aux_outputs:
            mri_struct_tokens = self._split_projected_features(mf_struct, ['t1', 't1c', 't2'])
            ct_struct_tokens = self._split_projected_features(cf_struct, ['ct', 'pet'])
            mri_response_tokens = self._split_projected_features(mf_response, ['t1', 't1c', 't2'])
            ct_response_tokens = self._split_projected_features(cf_response, ['ct', 'pet'])
            struct_features = {}
            response_features = {}
            for name in ['t1', 't1c', 't2', 'ct', 'pet']:
                present = present_dict[name].view(b, 1)
                if name in mri_struct_tokens:
                    struct_features[name] = mri_struct_tokens[name] * present
                    response_features[name] = mri_response_tokens[name] * present
                else:
                    struct_features[name] = ct_struct_tokens[name] * present
                    response_features[name] = ct_response_tokens[name] * present
            self.latest_aux = {'struct_features': struct_features, 'response_features': response_features, 'meta_feature': meta_raw_feat, 'present_dict': present_dict, 'fusion_weights': self.transformer_model.latest_fusion_weights, 'map_feature': map_repr, 'token_feature': token_repr, 'base_logits': base_logits, 'pet_residual_token': pet_residual_token, 'pet_residual_feature': pet_residual_feature, 'pet_residual_weight': pet_residual_weight, 'pet_map_descriptor': pet_map_descriptor, 'structure_input_mode': 'instance_norm_plus_gradient'}
        else:
            self.latest_aux = None
        return (logits, out_sequence, features_dict)
