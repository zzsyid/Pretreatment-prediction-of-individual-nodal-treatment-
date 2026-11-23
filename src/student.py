import torch
import torch.nn as nn
from robustness import convert_response_norm, student_mri_style
from teacher import AttentionPool, FamilyMapFusion, FeatProjector, make_structure_guided_input, reactExtractor, structExtractor

class CrossModalResponseBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.source_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, state, struct_condition, source_tokens, key_padding_mask=None):
        query = self.query_norm(state + struct_condition).unsqueeze(1)
        source_tokens = self.source_norm(source_tokens)
        attn_out, _ = self.attn(query, source_tokens, source_tokens, key_padding_mask=key_padding_mask)
        state = state + attn_out.squeeze(1)
        state = state + self.ffn(self.ffn_norm(state))
        return state

class StructureGuidedAnchorMatcher(nn.Module):

    def __init__(self, feature_dim, meta_dim, hidden_dim=256, target_names=('t1', 't1c', 't2', 'ct', 'pet')):
        super().__init__()
        self.scale = hidden_dim ** (-0.5)
        self.target_names = tuple(target_names)
        self.query_proj = nn.Sequential(nn.Linear(feature_dim + meta_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.key_proj = nn.Sequential(nn.Linear(feature_dim + meta_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.value_proj = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim))
        self.null_key = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.null_value = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.target_embeddings = nn.ParameterDict({name: nn.Parameter(torch.randn(1, hidden_dim) * 0.02) for name in self.target_names})

    def forward(self, target_struct, source_responses, source_masks, meta_feat, target_name='t1c'):
        b = target_struct.size(0)
        target_embed = self.target_embeddings[target_name].expand(b, -1)
        query = self.query_proj(torch.cat([target_struct, meta_feat], dim=1)) + target_embed
        query = query.unsqueeze(1)
        keys = []
        values = []
        for src in source_responses:
            keys.append(self.key_proj(torch.cat([src, meta_feat], dim=1)))
            values.append(self.value_proj(src))
        keys = torch.stack(keys, dim=1)
        values = torch.stack(values, dim=1)
        keys = torch.cat([keys, self.null_key.expand(b, -1, -1)], dim=1)
        values = torch.cat([values, self.null_value.expand(b, -1, -1)], dim=1)
        source_mask = torch.stack(source_masks, dim=1) < 0.5
        null_mask = torch.zeros((b, 1), dtype=torch.bool, device=source_mask.device)
        key_padding_mask = torch.cat([source_mask, null_mask], dim=1)
        logits = (query * keys).sum(dim=-1) * self.scale
        logits = logits.masked_fill(key_padding_mask, -10000.0)
        attn = torch.softmax(logits, dim=1)
        return (attn.unsqueeze(-1) * values).sum(dim=1)

class StructureGuidedResponseGenerator(nn.Module):

    def __init__(self, feature_dim, meta_dim, hidden_dim=256, num_heads=8, num_layers=6, target_names=('t1', 't1c', 't2', 'ct', 'pet')):
        super().__init__()
        self.target_names = tuple(target_names)
        self.struct_condition = nn.Sequential(nn.Linear(feature_dim + meta_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.source_adapter = nn.Sequential(nn.Linear(feature_dim + meta_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.anchor_adapter = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.flow_state_adapter = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.time_adapter = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.blocks = nn.ModuleList([CrossModalResponseBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.shared_out_proj = nn.Sequential(nn.Linear(hidden_dim, feature_dim), nn.LayerNorm(feature_dim))
        self.target_embeddings = nn.ParameterDict({name: nn.Parameter(torch.randn(1, hidden_dim) * 0.02) for name in self.target_names})
        self.target_output_heads = nn.ModuleDict({name: nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim)) for name in self.target_names})
        self.target_quality_heads = nn.ModuleDict({name: nn.Sequential(nn.Linear(feature_dim * 3 + meta_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()) for name in self.target_names})

    def forward(self, target_struct, source_responses, source_masks, meta_feat, anchor_response, flow_state=None, flow_time=None, target_name='t1c', return_aux=False):
        b = target_struct.size(0)
        if flow_state is None:
            flow_state = anchor_response
        if flow_time is None:
            flow_time = torch.zeros((b, 1), device=target_struct.device, dtype=target_struct.dtype)
        elif flow_time.dim() == 1:
            flow_time = flow_time.view(b, 1)
        target_embed = self.target_embeddings[target_name].expand(b, -1)
        struct_condition = self.struct_condition(torch.cat([target_struct, meta_feat], dim=1)) + target_embed
        anchor_hidden = self.anchor_adapter(anchor_response)
        flow_hidden = self.flow_state_adapter(flow_state)
        flow_time = flow_time.to(device=target_struct.device, dtype=target_struct.dtype)
        time_hidden = self.time_adapter(flow_time)
        state = flow_hidden + anchor_hidden + time_hidden + target_embed
        source_tokens = []
        for src in source_responses:
            source_tokens.append(self.source_adapter(torch.cat([src, meta_feat], dim=1)))
        source_tokens = torch.stack(source_tokens, dim=1)
        null_token = self.null_token.expand(b, -1, -1)
        source_tokens = torch.cat([source_tokens, null_token], dim=1)
        source_mask = torch.stack(source_masks, dim=1) < 0.5
        null_mask = torch.zeros((b, 1), dtype=torch.bool, device=source_mask.device)
        key_padding_mask = torch.cat([source_mask, null_mask], dim=1)
        for block in self.blocks:
            state = block(state, struct_condition, source_tokens, key_padding_mask=key_padding_mask)
        shared_velocity = self.shared_out_proj(state)
        velocity = shared_velocity + self.target_output_heads[target_name](shared_velocity)
        generated_response = flow_state + velocity
        quality = self.target_quality_heads[target_name](torch.cat([anchor_response, generated_response, velocity, meta_feat], dim=1))
        if not return_aux:
            return generated_response
        return (generated_response, {'anchor_response': anchor_response, 'flow_state': flow_state, 'flow_time': flow_time, 'generated_response': generated_response, 'velocity': velocity, 'quality': quality})

    @staticmethod
    def _masked_mean(tokens, masks):
        stacked_tokens = torch.stack(tokens, dim=1)
        stacked_masks = torch.stack(masks, dim=1).float()
        denom = stacked_masks.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (stacked_tokens * stacked_masks.unsqueeze(-1)).sum(dim=1) / denom

class PlainDualLevelTokenFusion(nn.Module):

    def __init__(self, cnn_feat_dim=128, meta_feat_dim=16, d_model=256, num_heads=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.proj_layers = nn.ModuleDict({'t1': nn.Linear(cnn_feat_dim, d_model), 't1c': nn.Linear(cnn_feat_dim, d_model), 't2': nn.Linear(cnn_feat_dim, d_model), 'ct': nn.Linear(cnn_feat_dim, d_model), 'pet': nn.Linear(cnn_feat_dim, d_model), 'meta': nn.Linear(meta_feat_dim, d_model)})
        self.modality_embs = nn.Parameter(torch.randn(6, 1, d_model) * 0.02)
        self.state_embeddings = nn.Embedding(3, d_model)
        nn.init.normal_(self.state_embeddings.weight, std=0.02)
        self.mask_tokens = nn.ParameterDict({'t1c': nn.Parameter(torch.randn(1, 1, d_model) * 0.02), 'pet': nn.Parameter(torch.randn(1, 1, d_model) * 0.02)})
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = AttentionPool(d_model)
        self.latest_fusion_weights = None
        self.latest_state_ids = None
        self.generated_priority = True

    def forward(self, features_dict, present_dict, meta_raw_feat, generated_dict=None):
        b = meta_raw_feat.size(0)
        seqs = []
        state_ids = {}
        for idx, name in enumerate(['t1', 't1c', 't2', 'ct', 'pet']):
            token = self.proj_layers[name](features_dict[name]).unsqueeze(1)
            present = present_dict[name].view(b)
            generated = generated_dict[name].to(device=present.device).view(b) > 0.5 if generated_dict is not None and name in generated_dict else torch.zeros_like(present, dtype=torch.bool)
            state_id = torch.where(generated & ((present <= 0.5) | self.generated_priority), torch.ones_like(present, dtype=torch.long), torch.where(present > 0.5, torch.zeros_like(present, dtype=torch.long), torch.full_like(present, 2, dtype=torch.long)))
            state_ids[name] = state_id
            if name in self.mask_tokens:
                mask = present.view(b, 1, 1).expand(-1, -1, self.d_model)
                token = torch.where(mask > 0.5, token, self.mask_tokens[name].expand(b, -1, -1))
            seqs.append(token + self.modality_embs[idx] + self.state_embeddings(state_id).unsqueeze(1))
        meta_token = self.proj_layers['meta'](meta_raw_feat).unsqueeze(1) + self.modality_embs[5]
        out_sequence = self.transformer(torch.cat(seqs + [meta_token], dim=1))
        self.latest_fusion_weights = None
        self.latest_state_ids = state_ids
        return (self.attention_pool(out_sequence), out_sequence)

class SimpleFamilyMapFusion(nn.Module):

    def __init__(self, channels, out_dim):
        super().__init__()
        self.conv = nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(channels, out_dim)

    def forward(self, struct_map, response_map):
        fused = self.conv(torch.cat([struct_map, response_map], dim=1))
        pooled = self.pool(fused).flatten(1)
        return (self.proj(pooled), fused)

class SimpleTokenFuse(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()
        self.proj = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, struct_feat, response_feat):
        return self.proj(torch.cat([struct_feat, response_feat], dim=1))

class PETQueryResponseGenerator(nn.Module):

    def __init__(self, feature_dim, meta_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.query_proj = nn.Sequential(nn.Linear(feature_dim * 2 + meta_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.source_proj = nn.Sequential(nn.Linear(feature_dim * 2 + meta_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.source_embeddings = nn.Parameter(torch.randn(4, 1, hidden_dim) * 0.02)
        self.null_source = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.source_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.delta_head = nn.Sequential(nn.Linear(hidden_dim, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        self.feature_gate_head = nn.Sequential(nn.Linear(hidden_dim, feature_dim), nn.Sigmoid())
        self.quality_head = nn.Sequential(nn.Linear(hidden_dim + feature_dim * 2 + meta_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, ct_response, ct_struct, source_structs, source_responses, source_masks, meta_feat, ct_present, return_aux=False):
        b = ct_response.size(0)
        query = self.query_proj(torch.cat([ct_response, ct_struct, meta_feat], dim=1)).unsqueeze(1)
        source_tokens = []
        for idx, (source_struct, source_response) in enumerate(zip(source_structs, source_responses)):
            source = self.source_proj(torch.cat([source_struct, source_response, meta_feat], dim=1)).unsqueeze(1)
            source_tokens.append(source + self.source_embeddings[idx])
        source_tokens = torch.cat(source_tokens, dim=1)
        source_tokens = torch.cat([source_tokens, self.null_source.expand(b, -1, -1)], dim=1)
        source_mask = torch.stack(source_masks, dim=1) < 0.5
        null_mask = torch.zeros((b, 1), dtype=torch.bool, device=source_mask.device)
        key_padding_mask = torch.cat([source_mask, null_mask], dim=1)
        normalized_sources = self.source_norm(source_tokens)
        attended, attention_weights = self.cross_attention(self.query_norm(query), normalized_sources, normalized_sources, key_padding_mask=key_padding_mask)
        hidden = (query + attended).squeeze(1)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        residual = self.delta_head(hidden)
        feature_gate = self.feature_gate_head(hidden)
        quality = self.quality_head(torch.cat([hidden, ct_response, ct_struct, meta_feat], dim=1))
        ct_mask = ct_present.view(-1, 1)
        generated_response = ct_response + ct_mask * quality * feature_gate * residual
        if not return_aux:
            return generated_response
        return (generated_response, {'anchor_response': ct_response, 'generated_response': generated_response, 'velocity': residual, 'feature_gate': feature_gate, 'quality': quality, 'ct_present': ct_mask, 'attention_weights': attention_weights, 'generator_type': 'pet_query_cross_attention'})

class CounterfactualPETEffectAdapter(nn.Module):

    def __init__(self, feature_dim, token_dim, map_dim, meta_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.proxy_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.generated_state_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.source_norm = nn.LayerNorm(token_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.delta_logit_head = nn.Sequential(nn.Linear(hidden_dim + token_dim + map_dim + meta_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, pet_proxy_feature, base_token_repr, base_sequence, map_repr, meta_raw_feat, quality, effect_mask, source_present_dict, context_mode='full'):
        b = pet_proxy_feature.size(0)
        query = self.proxy_proj(pet_proxy_feature).unsqueeze(1) + self.generated_state_embedding
        source_mask = torch.stack([source_present_dict[name].view(b) < 0.5 for name in ('t1', 't1c', 't2', 'ct', 'pet')], dim=1)
        meta_mask = torch.zeros((b, 1), dtype=torch.bool, device=source_mask.device)
        key_padding_mask = torch.cat([source_mask, meta_mask], dim=1)
        normalized_sources = self.source_norm(base_sequence)
        attended, attention_weights = self.cross_attention(self.query_norm(query), normalized_sources, normalized_sources, key_padding_mask=key_padding_mask)
        if context_mode == 'innovation':
            effect_feature = (query - attended).squeeze(1)
            head_base_token = torch.zeros_like(base_token_repr)
            head_map = torch.zeros_like(map_repr)
            head_meta = torch.zeros_like(meta_raw_feat)
        elif context_mode == 'full':
            effect_feature = (query + attended).squeeze(1)
            head_base_token = base_token_repr
            head_map = map_repr
            head_meta = meta_raw_feat
        else:
            raise ValueError(f'Unsupported PET effect context mode: {context_mode}')
        effect_feature = effect_feature + self.ffn(self.ffn_norm(effect_feature))
        raw_delta_logit = self.delta_logit_head(torch.cat([effect_feature, head_base_token, head_map, head_meta], dim=1))
        effect_weight = quality.clamp(0.0, 1.0) * effect_mask.view(-1, 1)
        delta_logit = raw_delta_logit * effect_weight
        effect_token = effect_feature * effect_weight
        return (delta_logit, effect_token, effect_feature, effect_weight, raw_delta_logit, attention_weights)

class MultiModalLymphNodeNet(nn.Module):

    def __init__(self, num_regions=15, feature_dim=128, disable_dual_branch_fusion=False, disable_pet_aware_modeling=False, disable_missing_modality_generation=False, mri_response_norm='instance', mri_style_augmentation=True, generated_token_priority=True):
        super().__init__()
        inputdim = 96
        region_emb_dim = 8
        meta_feature_dim = 16
        self.feature_dim = feature_dim
        self.fd = feature_dim
        self.modality_names = ('t1', 't1c', 't2', 'ct', 'pet')
        self.structure_gradient_weight = 0.5
        self.disable_dual_branch_fusion = bool(disable_dual_branch_fusion)
        self.disable_pet_aware_modeling = bool(disable_pet_aware_modeling)
        self.disable_missing_modality_generation = bool(disable_missing_modality_generation)
        self.pet_effect_mode = 'single_gate'
        self.pet_effect_context_mode = 'full'
        self.pet_effect_scale = 1.0
        self.mri_response_norm = mri_response_norm
        self.mri_style_augmentation = bool(mri_style_augmentation)
        self.region_emb = nn.Embedding(num_regions, region_emb_dim)
        self.strct_extractor_mri = structExtractor(in_channels=1, feature_dim=feature_dim, dim=inputdim, fs=3)
        self.strct_extractor_ct = structExtractor(in_channels=1, feature_dim=feature_dim, dim=inputdim, fs=2)
        self.react_extractor_mri = reactExtractor(in_channels=1, feature_dim=feature_dim, dim=inputdim, fs=3)
        if mri_response_norm == 'instance':
            assert convert_response_norm(self.react_extractor_mri) == 6
        elif mri_response_norm != 'batch':
            raise ValueError('Unknown MRI response normalization: ' + mri_response_norm)
        self.react_extractor_ct = reactExtractor(in_channels=1, feature_dim=feature_dim, dim=inputdim, fs=2)
        self.mp = FeatProjector(1, feature_dim, dim=inputdim, fs=3)
        self.cp = FeatProjector(1, feature_dim, dim=inputdim, fs=2)
        map_fusion_cls = SimpleFamilyMapFusion if self.disable_dual_branch_fusion else FamilyMapFusion
        self.mri_map_fusion = map_fusion_cls(channels=inputdim * 2 * 3, out_dim=feature_dim)
        self.ct_map_fusion = map_fusion_cls(channels=inputdim * 2 * 2, out_dim=feature_dim)
        self.map_joint_fuser = nn.Sequential(nn.Linear(feature_dim * 2, 256), nn.GELU(), nn.LayerNorm(256))
        meta_input_dim = 1 + region_emb_dim
        self.meta_encoder = nn.Sequential(nn.Linear(meta_input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=False), nn.Linear(32, meta_feature_dim), nn.ReLU(inplace=False))
        if self.disable_dual_branch_fusion:
            self.feature_fusers = nn.ModuleDict({name: SimpleTokenFuse(feature_dim) for name in ['t1', 't1c', 't2', 'ct', 'pet']})
        else:
            self.feature_fusers = nn.ModuleDict({name: nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim)) for name in ['t1', 't1c', 't2', 'ct', 'pet']})
        self.mri_from_ct_struct_adapter = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        self.ct_from_mri_struct_adapter = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        self.mri_target_struct_adapters = nn.ModuleDict({name: nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim)) for name in ['t1', 't1c', 't2']})
        self.ct_target_struct_adapter = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        self.anchor_matcher = StructureGuidedAnchorMatcher(feature_dim, meta_feature_dim, target_names=self.modality_names)
        self.response_generator = StructureGuidedResponseGenerator(feature_dim, meta_feature_dim, target_names=self.modality_names)
        self.pet_response_generator = PETQueryResponseGenerator(feature_dim, meta_feature_dim)
        self.t1c_response_generator = self.response_generator
        self.transformer_model = PlainDualLevelTokenFusion(cnn_feat_dim=feature_dim, meta_feat_dim=meta_feature_dim, d_model=256, num_heads=4, num_layers=2)
        self.pet_effect_adapter = CounterfactualPETEffectAdapter(feature_dim=feature_dim, token_dim=256, map_dim=256, meta_dim=meta_feature_dim, hidden_dim=256)
        self.simple_missing_response_adapters = nn.ModuleDict({name: nn.Linear(feature_dim, feature_dim) for name in self.modality_names})
        self.classifier = nn.Sequential(nn.Linear(256 + 256 + meta_feature_dim, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))
        self.enable_aux_outputs = False
        self.latest_aux = None
        self.transformer_model.generated_priority = bool(generated_token_priority)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        optional_prefixes = (f'{prefix}pet_proxy_residual_branch.', f'{prefix}pet_effect_adapter.', f'{prefix}pet_response_generator.', f'{prefix}simple_missing_response_adapters.')
        missing_keys[:] = [key for key in missing_keys if not key.startswith(optional_prefixes)]
        unexpected_prefixes = (f'{prefix}pet_struct_adapter.', f'{prefix}pet_proxy_residual_branch.')
        unexpected_keys[:] = [key for key in unexpected_keys if not key.startswith(unexpected_prefixes)]

    def _split_projected_features(self, projected_feat, names):
        output = {}
        for i, name in enumerate(names):
            output[name] = projected_feat[:, i * self.fd:(i + 1) * self.fd]
        return output

    @staticmethod
    def _masked_mean(tokens, masks):
        stacked_tokens = torch.stack(tokens, dim=1)
        stacked_masks = torch.stack(masks, dim=1).float()
        denom = stacked_masks.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (stacked_tokens * stacked_masks.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _mask_feature(feature, present):
        return feature * present.float().view(feature.size(0), 1)

    @staticmethod
    def _present_or_default(present_dict, name, batch_size, device, dtype, default):
        if present_dict is not None and name in present_dict:
            return present_dict[name].to(device=device, dtype=dtype).view(batch_size)
        return torch.full((batch_size,), float(default), device=device, dtype=dtype)

    @staticmethod
    def _zero_like_channel(reference):
        return torch.zeros_like(reference[:, 0:1, ...])

    def _fuse_feature(self, name, struct_feat, response_feat):
        if self.disable_dual_branch_fusion:
            return self.feature_fusers[name](struct_feat, response_feat)
        return self.feature_fusers[name](torch.cat([struct_feat, response_feat], dim=1))

    def _source_lists_for_target(self, target_name, response_features, present_dict):
        source_names = [name for name in self.modality_names if name != target_name]
        source_responses = [response_features[name] for name in source_names]
        source_masks = [present_dict[name] for name in source_names]
        return (source_responses, source_masks)

    def _generate_target_response(self, target_name, target_struct, struct_features, response_features, present_dict, meta_raw_feat):
        if self.disable_missing_modality_generation:
            generated_response = self.simple_missing_response_adapters[target_name](target_struct)
            return (generated_response, {'generator_type': 'linear_substitute', 'target_name': target_name, 'target_struct': target_struct})
        source_responses, source_masks = self._source_lists_for_target(target_name, response_features, present_dict)
        if target_name == 'pet':
            mri_struct_context = self._masked_mean([struct_features['t1'], struct_features['t1c'], struct_features['t2']], [present_dict['t1'], present_dict['t1c'], present_dict['t2']])
            mri_response_context = self._masked_mean([response_features['t1'], response_features['t1c'], response_features['t2']], [present_dict['t1'], present_dict['t1c'], present_dict['t2']])
            fallback_anchor = self.anchor_matcher(target_struct=target_struct, source_responses=source_responses, source_masks=source_masks, meta_feat=meta_raw_feat, target_name=target_name)
            generic_generated_response, generic_aux = self.response_generator(target_struct=target_struct, source_responses=source_responses, source_masks=source_masks, meta_feat=meta_raw_feat, anchor_response=fallback_anchor, target_name=target_name, return_aux=True)
            if self.disable_pet_aware_modeling:
                generated_response = generic_generated_response
                generator_aux = dict(generic_aux)
                generator_aux.update({'fallback_anchor': fallback_anchor, 'fallback_generated_response': generic_generated_response, 'generator_type': 'generic_pet_generator', 'target_name': target_name, 'target_struct': target_struct, 'source_responses': source_responses, 'source_masks': source_masks, 'meta_feat': meta_raw_feat, 'mri_struct_context': mri_struct_context, 'mri_response_context': mri_response_context})
            else:
                pet_source_names = ('t1', 't1c', 't2', 'ct')
                ct_generated_response, generator_aux = self.pet_response_generator(ct_response=response_features['ct'], ct_struct=struct_features['ct'], source_structs=[struct_features[name] for name in pet_source_names], source_responses=[response_features[name] for name in pet_source_names], source_masks=[present_dict[name] for name in pet_source_names], meta_feat=meta_raw_feat, ct_present=present_dict['ct'], return_aux=True)
                use_ct_transform = (present_dict['ct'] > 0.5).view(-1, 1)
                generated_response = torch.where(use_ct_transform, ct_generated_response, generic_generated_response)
                generator_aux['quality'] = torch.where(use_ct_transform, generator_aux['quality'], generic_aux['quality'])
                generator_aux['generated_response'] = generated_response
                generator_aux.update({'fallback_anchor': fallback_anchor, 'fallback_generated_response': generic_generated_response, 'generator': self.pet_response_generator, 'target_name': target_name, 'target_struct': target_struct, 'source_responses': source_responses, 'source_masks': source_masks, 'meta_feat': meta_raw_feat, 'mri_struct_context': mri_struct_context, 'mri_response_context': mri_response_context})
            return (generated_response, generator_aux)
        anchor_response = self.anchor_matcher(target_struct=target_struct, source_responses=source_responses, source_masks=source_masks, meta_feat=meta_raw_feat, target_name=target_name)
        generated_response, generator_aux = self.response_generator(target_struct=target_struct, source_responses=source_responses, source_masks=source_masks, meta_feat=meta_raw_feat, anchor_response=anchor_response, target_name=target_name, return_aux=True)
        generator_aux.update({'generator': self.response_generator, 'target_name': target_name, 'target_struct': target_struct, 'source_responses': source_responses, 'source_masks': source_masks, 'meta_feat': meta_raw_feat})
        return (generated_response, generator_aux)

    def forward(self, mri_imgs, mri_present, ct_pet_imgs, ct_present, vols, regions):
        if self.training and self.mri_style_augmentation:
            mri_imgs = student_mri_style(mri_imgs)
        b = mri_imgs.size(0)
        device = mri_imgs.device
        dtype = mri_imgs.dtype
        present_dict = {'t1': self._present_or_default(mri_present, 't1', b, device, dtype, 1.0), 't1c': self._present_or_default(mri_present, 't1c', b, device, dtype, 0.0), 't2': self._present_or_default(mri_present, 't2', b, device, dtype, 1.0), 'ct': self._present_or_default(ct_present, 'ct', b, device, dtype, 0.0), 'pet': self._present_or_default(ct_present, 'pet', b, device, dtype, 0.0)}
        if mri_imgs.size(1) == 2:
            t1_raw = mri_imgs[:, 0:1, ...]
            t1c_raw = self._zero_like_channel(mri_imgs)
            t2_raw = mri_imgs[:, 1:2, ...]
            present_dict['t1c'] = torch.zeros_like(present_dict['t1c'])
        else:
            t1_raw = mri_imgs[:, 0:1, ...]
            t1c_raw = mri_imgs[:, 1:2, ...] if mri_imgs.size(1) > 1 else self._zero_like_channel(mri_imgs)
            t2_raw = mri_imgs[:, 2:3, ...] if mri_imgs.size(1) > 2 else self._zero_like_channel(mri_imgs)
        if ct_pet_imgs is None:
            ct_raw = torch.zeros_like(t1_raw)
            pet_raw = torch.zeros_like(t1_raw)
            present_dict['ct'] = torch.zeros_like(present_dict['ct'])
            present_dict['pet'] = torch.zeros_like(present_dict['pet'])
        else:
            ct_pet_imgs = ct_pet_imgs.to(device=device, dtype=dtype)
            if ct_pet_imgs.size(1) == 0:
                ct_raw = torch.zeros_like(t1_raw)
                pet_raw = torch.zeros_like(t1_raw)
            else:
                ct_raw = ct_pet_imgs[:, 0:1, ...]
                pet_raw = ct_pet_imgs[:, 1:2, ...] if ct_pet_imgs.size(1) > 1 else torch.zeros_like(ct_raw)
                if ct_pet_imgs.size(1) == 1:
                    present_dict['pet'] = torch.zeros_like(present_dict['pet'])
        t1 = t1_raw * present_dict['t1'].view(b, 1, 1, 1, 1)
        t1c = t1c_raw * present_dict['t1c'].view(b, 1, 1, 1, 1)
        t2 = t2_raw * present_dict['t2'].view(b, 1, 1, 1, 1)
        ct = ct_raw * present_dict['ct'].view(b, 1, 1, 1, 1)
        pet = pet_raw * present_dict['pet'].view(b, 1, 1, 1, 1)
        mriimages = torch.cat((t1, t1c, t2), dim=1)
        ctimages = torch.cat((ct, pet), dim=1)
        mri_struct_images = make_structure_guided_input(mriimages, self.structure_gradient_weight)
        ct_struct_images = make_structure_guided_input(ctimages, self.structure_gradient_weight)
        mrisf = self.strct_extractor_mri(mri_struct_images)
        ctsf = self.strct_extractor_ct(ct_struct_images)
        mrirf = self.react_extractor_mri(mriimages)
        ctrf = self.react_extractor_ct(ctimages)
        mri_struct_tokens = self._split_projected_features(self.mp(mrisf), ['t1', 't1c', 't2'])
        ct_struct_tokens = self._split_projected_features(self.cp(ctsf), ['ct', 'pet'])
        mri_response_tokens = self._split_projected_features(self.mp(mrirf), ['t1', 't1c', 't2'])
        ct_response_tokens = self._split_projected_features(self.cp(ctrf), ['ct', 'pet'])
        struct_features = {'t1': mri_struct_tokens['t1'], 't1c': mri_struct_tokens['t1c'], 't2': mri_struct_tokens['t2'], 'ct': ct_struct_tokens['ct'], 'pet': ct_struct_tokens['pet']}
        response_features = {'t1': mri_response_tokens['t1'], 't1c': mri_response_tokens['t1c'], 't2': mri_response_tokens['t2'], 'ct': ct_response_tokens['ct'], 'pet': ct_response_tokens['pet']}
        reg_emb = self.region_emb(regions.long().view(b))
        meta_input = torch.cat([vols.view(b, 1), reg_emb], dim=1)
        meta_raw_feat = self.meta_encoder(meta_input)
        mri_family_present = torch.maximum(present_dict['t1'], present_dict['t2'])
        mri_shared_struct = self._masked_mean([struct_features['t1'], struct_features['t2']], [present_dict['t1'], present_dict['t2']])
        ct_observed_struct = self._mask_feature(struct_features['ct'], present_dict['ct'])
        mri_struct_from_ct = self.mri_from_ct_struct_adapter(ct_observed_struct)
        mri_base_struct = torch.where((mri_family_present > 0.5).view(b, 1), mri_shared_struct, mri_struct_from_ct)
        ct_struct_from_mri = self.ct_from_mri_struct_adapter(mri_base_struct)
        ct_base_struct = torch.where((present_dict['ct'] > 0.5).view(b, 1), struct_features['ct'], ct_struct_from_mri)
        shared_ct_struct = self.ct_target_struct_adapter(ct_base_struct)
        target_structs = {'t1': self.mri_target_struct_adapters['t1'](mri_base_struct), 't1c': self.mri_target_struct_adapters['t1c'](mri_base_struct), 't2': self.mri_target_struct_adapters['t2'](mri_base_struct), 'ct': shared_ct_struct, 'pet': shared_ct_struct}
        generated_aux = {}
        generated_names = []
        for name in self.modality_names:
            missing = (present_dict[name] < 0.5).view(b, 1)
            if not bool(missing.any()):
                continue
            generated_response, generator_aux = self._generate_target_response(target_name=name, target_struct=target_structs[name], struct_features=struct_features, response_features=response_features, present_dict=present_dict, meta_raw_feat=meta_raw_feat)
            if name == 'pet' and self.pet_effect_mode in {'single_gate', 'unit_quality'} and (generator_aux.get('generator_type') == 'pet_query_cross_attention'):
                generated_response = generator_aux['anchor_response'] + generator_aux['ct_present'] * generator_aux['feature_gate'] * generator_aux['velocity']
                generator_aux['generated_response'] = generated_response
            struct_features[name] = torch.where(missing, target_structs[name], struct_features[name])
            response_features[name] = torch.where(missing, generated_response, response_features[name])
            generated_aux[name] = generator_aux
            generated_names.append(name)
        features_dict = {}
        for name in ['t1', 't1c', 't2', 'ct', 'pet']:
            features_dict[name] = self._fuse_feature(name, struct_features[name], response_features[name])
        mri_map_repr, _ = self.mri_map_fusion(mrisf, mrirf)
        ct_map_repr, _ = self.ct_map_fusion(ctsf, ctrf)
        ct_family_present = torch.maximum(present_dict['ct'], present_dict['pet']).view(b, 1)
        mri_map_repr = mri_map_repr * mri_family_present.view(b, 1)
        ct_map_repr = ct_map_repr * ct_family_present
        map_repr = self.map_joint_fuser(torch.cat([mri_map_repr, ct_map_repr], dim=1))
        effective_present_dict = dict(present_dict)
        generated_state_dict = {name: torch.zeros_like(present_dict[name]) for name in self.modality_names}
        for name in generated_names:
            effective_present_dict[name] = torch.ones_like(present_dict[name])
            generated_state_dict[name] = (present_dict[name] < 0.5).to(dtype=dtype)
        base_present_dict = dict(effective_present_dict)
        base_present_dict['pet'] = present_dict['pet']
        token_repr, out_sequence = self.transformer_model(features_dict, base_present_dict, meta_raw_feat, generated_dict=generated_state_dict)
        pet_proxy_delta_response = response_features['pet'] - response_features['ct']
        pet_proxy_mask = (present_dict['pet'] < 0.5).to(dtype=dtype) * (present_dict['ct'] > 0.5).to(dtype=dtype)
        if self.disable_pet_aware_modeling:
            pet_proxy_mask = torch.zeros_like(pet_proxy_mask)
        pet_generator_aux = generated_aux.get('pet', {})
        pet_quality = pet_generator_aux.get('quality', torch.zeros((b, 1), device=device, dtype=dtype))
        adapter_quality = torch.ones_like(pet_quality) if self.pet_effect_mode == 'unit_quality' else pet_quality
        pet_effect_logit, pet_effect_token, pet_proxy_feature, pet_effect_weight, pet_effect_raw_logit, pet_effect_attention = self.pet_effect_adapter(pet_proxy_feature=features_dict['pet'], base_token_repr=token_repr, base_sequence=out_sequence, map_repr=map_repr, meta_raw_feat=meta_raw_feat, quality=adapter_quality, effect_mask=pet_proxy_mask, source_present_dict=base_present_dict, context_mode=self.pet_effect_context_mode)
        if self.pet_effect_mode == 'zero_effect':
            pet_effect_logit = torch.zeros_like(pet_effect_logit)
            pet_effect_token = torch.zeros_like(pet_effect_token)
            pet_effect_weight = torch.zeros_like(pet_effect_weight)
        base_logits = self.classifier(torch.cat([map_repr, token_repr, meta_raw_feat], dim=1))
        logits = base_logits + float(self.pet_effect_scale) * pet_effect_logit
        for generator_aux in generated_aux.values():
            quality = generator_aux.get('quality')
            if quality is not None:
                logits = logits + quality.sum(dim=1, keepdim=True) * 0.0
        if self.enable_aux_outputs:
            for name in ['t1', 't1c', 't2', 'ct', 'pet']:
                present = effective_present_dict[name].view(b, 1)
                struct_features[name] = struct_features[name] * present
                response_features[name] = response_features[name] * present
            self.latest_aux = {'struct_features': struct_features, 'response_features': response_features, 'meta_feature': meta_raw_feat, 'present_dict': effective_present_dict, 'observed_present_dict': present_dict, 'base_present_dict': base_present_dict, 'generated_state_dict': generated_state_dict, 'modality_state_ids': self.transformer_model.latest_state_ids, 'fusion_weights': self.transformer_model.latest_fusion_weights, 'meanflow_features': generated_aux, 'map_feature': map_repr, 'token_feature': token_repr, 'base_logits': base_logits, 'pet_proxy_delta_response': pet_proxy_delta_response, 'pet_proxy_token': pet_proxy_feature, 'pet_proxy_residual_token': pet_effect_token, 'pet_proxy_residual_feature': pet_proxy_feature, 'pet_proxy_residual_weight': pet_effect_weight, 'pet_effect_token': pet_effect_token, 'pet_effect_logit': pet_effect_logit, 'pet_effect_raw_logit': pet_effect_raw_logit, 'pet_effect_weight': pet_effect_weight, 'pet_effect_attention': pet_effect_attention, 'pet_generation_quality': pet_quality, 'pet_proxy_mask': pet_proxy_mask, 'structure_input_mode': 'instance_norm_plus_gradient'}
        else:
            self.latest_aux = None
        return (logits, out_sequence, features_dict)
