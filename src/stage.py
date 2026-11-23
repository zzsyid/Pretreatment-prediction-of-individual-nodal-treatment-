import argparse
import gc
import hashlib
import inspect
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from common import RUN

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_torch(path, value):
    temp = str(path) + '.tmp'
    torch.save(value, temp)
    os.replace(temp, str(path))

class ProgressLoader:

    def __init__(self, loader, stage, epoch, phase):
        self.loader, self.stage, self.epoch, self.phase = (loader, stage, epoch, phase)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        start = time.monotonic()
        for index, batch in enumerate(self.loader, 1):
            if index == 1 or index % 20 == 0 or index == len(self):
                elapsed = time.monotonic() - start
                write(RUN / 'worker_status.json', {'stage': self.stage, 'epoch': self.epoch + 1, 'phase': self.phase, 'batch': index, 'batches': len(self), 'elapsed_seconds': elapsed, 'eta_epoch_phase_seconds': elapsed / index * (len(self) - index), 'pid': os.getpid(), 'updated_utc': datetime.now(timezone.utc).isoformat()})
            yield batch

    def __getattr__(self, name):
        return getattr(self.loader, name)

def keyword_call(function, positional, values):
    signature = inspect.signature(function)
    occupied = signature.bind_partial(*positional).arguments
    kwargs = {name: values[name] for name in signature.parameters if name not in occupied and name in values}
    signature.bind(*positional, **kwargs)
    return function(*positional, **kwargs)

def main():
    wrapper = argparse.ArgumentParser(allow_abbrev=False)
    wrapper.add_argument('--source', choices=['warmup', 'adaptation'], required=True)
    wrapper.add_argument('--stage-key', required=True)
    wrapper.add_argument('--preflight', action='store_true')
    own, rest = wrapper.parse_known_args()
    physical = own.source == 'adaptation'
    import engine as training
    from losses import build_classification_loss
    sys.argv = ['stage.py'] + rest
    args = training.parse_args()
    args.physical_preprocessing = physical
    args.disable_balanced_sampler = True
    args.input_mode = training.resolve_input_mode(args.stage, args.model_impl, args.input_mode)
    training.set_random_seed(args.seed, deterministic=args.deterministic)
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('ANON_MEMORY_FRACTION', '.35')), 0)
    device = torch.device('cuda:0')
    protocol = json.loads((RUN / 'protocol.json').read_text())
    expected_hash = os.environ.get('ANON_PROTOCOL_SHA256')
    if expected_hash:
        assert sha(RUN / 'protocol.json') == expected_hash
    for split, meta in json.loads((RUN / 'split_report.json').read_text())['splits'].items():
        assert sha(meta['path']) == meta['sha256']
    save_dir = RUN / 'checkpoints' / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    if own.preflight:
        fixture = RUN / 'audit/preflight_train.json'
        args.train_json = str(fixture)
        args.val_json = str(fixture)
        args.train_cache_rate = args.eval_cache_rate = 0.0
        args.loader_workers = 0
        args.cache_workers = 1
    train_loader, _ = training.build_loader(args.train_json, args.batch_size, True, False, args.data_stage, args)
    val_loader, _ = training.build_loader(args.val_json, args.batch_size, False, False, args.data_stage, args)
    assert len(train_loader.dataset) % args.batch_size != 1, 'Training remainder would violate BatchNorm; do not discard patients silently'
    criterion = build_classification_loss(loss_name=args.loss, pos_weight=training.calculate_pos_weight_from_json(args.train_json, args.data_stage).to(device), focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha, asl_gamma_pos=args.asl_gamma_pos, asl_gamma_neg=args.asl_gamma_neg, asl_clip=args.asl_clip).to(device)
    effective = training.get_effective_weights(args)
    teacher_class, student_class = training.get_model_classes(args.model_impl)
    teacher = None
    parents = {}

    def load(model, path):
        path = Path(path).resolve()
        assert str(path).startswith(str(RUN / 'checkpoints') + '/')
        if not own.preflight:
            assert path.is_file()
            parents[str(path)] = sha(path)
            model = training.load_checkpoint_state(model, str(path), device)
        return model
    if args.stage == 'teacher':
        model = teacher_class(num_regions=args.num_regions, feature_dim=args.feature_dim, mri_response_norm='instance' if physical else 'batch', mri_style_augmentation=physical).to(device)
        if getattr(args, 'teacher_init_ckpt', None):
            model = load(model, args.teacher_init_ckpt)
    else:
        assert args.teacher_ckpt
        teacher = teacher_class(num_regions=args.num_regions, feature_dim=args.feature_dim, mri_response_norm='instance' if physical else 'batch', mri_style_augmentation=physical).to(device)
        teacher = load(training.disable_inplace(teacher), args.teacher_ckpt)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        model = student_class(num_regions=args.num_regions, feature_dim=args.feature_dim, disable_dual_branch_fusion=args.disable_dual_branch_fusion, disable_pet_aware_modeling=args.disable_pet_aware_modeling, disable_missing_modality_generation=args.disable_missing_modality_generation, mri_response_norm='instance' if physical else 'batch', mri_style_augmentation=physical, generated_token_priority=physical).to(device)
        if args.student_init_ckpt:
            model = load(model, args.student_init_ckpt)
        if hasattr(model, 'pet_effect_mode'):
            model.pet_effect_mode, model.pet_effect_context_mode = (args.pet_effect_mode, args.pet_effect_context_mode)
        if args.freeze_student_base:
            training.configure_cpea_stage2(model, zero_init_effect_head=args.zero_init_pet_effect_head)
    model = training.disable_inplace(model)
    training.set_aux_outputs_enabled(model, True)
    if teacher is not None:
        training.set_aux_outputs_enabled(teacher, True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    hidden_checks = {'student_forward_calls': 0}
    if own.preflight and args.stage == 'student':

        def check_hidden_inputs(module, inputs):
            mri, mri_present, ct_pet, ct_present = inputs[:4]
            assert torch.count_nonzero(mri[:, 1]).item() == 0, 'Student received CE-T1 pixels'
            assert torch.count_nonzero(ct_pet[:, 1]).item() == 0, 'Student received PET pixels'
            if isinstance(mri_present, torch.Tensor):
                assert torch.count_nonzero(mri_present[:, 1]).item() == 0
            elif isinstance(mri_present, dict):
                assert torch.count_nonzero(mri_present['t1c']).item() == 0
            if isinstance(ct_present, torch.Tensor):
                assert torch.count_nonzero(ct_present[:, 1]).item() == 0
            elif isinstance(ct_present, dict):
                assert torch.count_nonzero(ct_present['pet']).item() == 0
            hidden_checks['student_forward_calls'] += 1
        model.register_forward_pre_hook(check_hidden_inputs)
    config = {'stage': own.stage_key, 'source': own.source, 'args': vars(args), 'effective_weights': effective, 'parents_sha256': parents, 'preflight': own.preflight, 'protocol_sha256': expected_hash, 'model_trainable_parameters': sum((p.numel() for p in model.parameters() if p.requires_grad)), 'old_trained_weights_loaded': False, 'initialization_seed': args.seed}
    if not own.preflight:
        write(save_dir / 'config.json', config)
    epoch_start, best_value, best_epoch, bad_epochs, history = (0, -1e+18, -1, 0, [])
    resumed = False
    last_path, best_path = (save_dir / 'last.pth', save_dir / 'best.pth')
    if last_path.exists() and (not own.preflight):
        state = torch.load(str(last_path), map_location=device)
        assert state['stage'] == own.stage_key and state['protocol_sha256'] == expected_hash
        assert state['parents_sha256'] == parents
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch_start, best_value, best_epoch = (state['epoch'] + 1, state['best_value'], state['best_epoch'])
        bad_epochs, history = (state['bad_epochs'], state['history'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), 0)
        resumed = True
        print('Resuming', own.stage_key, 'at epoch', epoch_start + 1, flush=True)

    def save_last(epoch):
        save_torch(last_path, {'stage': own.stage_key, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'best_value': best_value, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'history': history, 'protocol_sha256': expected_hash, 'parents_sha256': parents, 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(0)})

    def validate(epoch):
        return training.validate_strict(model, ProgressLoader(val_loader, own.stage_key, epoch, 'validation'), criterion, device, args.stage, args.input_mode, args.disable_meta, student_hide_prob_t1c=args.student_hide_prob_t1c, student_hide_prob_pet=args.student_hide_prob_pet)

    def update_best(epoch, metrics):
        nonlocal best_value, best_epoch, bad_epochs
        improved = metrics['auc'] > best_value + args.early_stop_min_delta
        if improved:
            best_value, best_epoch, bad_epochs = (float(metrics['auc']), epoch, 0)
            save_torch(best_path, model.state_dict())
            write(save_dir / 'best_metrics.json', {'best_epoch': epoch, 'best_auc': best_value, 'best_threshold': metrics['threshold'], 'validation': metrics, 'selection': 'New validation patients only'})
        else:
            bad_epochs += 1
        return improved
    if getattr(args, 'validate_before_train', False) and (not resumed) and (not own.preflight):
        initial = validate(-1)
        update_best(-1, initial)
        history.append({'epoch': -1, 'role': 'initial_validation', **initial})
        write(save_dir / 'history.json', history)
        save_last(-1)
    if own.preflight:
        assert args.batch_size <= len(train_loader.dataset)
        original = train_loader

        class FirstBatch:

            def __len__(self):
                return 1

            def __iter__(self):
                yield next(iter(original))
        train_loader = FirstBatch()
        for group in optimizer.param_groups:
            group['lr'] = 0.0
    cap = 1 if own.preflight else args.epochs
    if not own.preflight and bad_epochs >= args.early_stop_patience > 0:
        epoch_start = cap
    for epoch in range(epoch_start, cap):
        started = time.monotonic()
        values = vars(args).copy()
        values.update(effective)
        hide_t1ce, hide_pet = training.get_student_hide_probs(args, epoch)
        values.update(hide_prob_t1c=hide_t1ce, hide_prob_pet=hide_pet, generated_response_kd_weight=training.get_warmup_weight(effective['generated_response_kd_weight'], epoch, args.generated_response_warmup_epochs), meanflow_response_weight=training.get_warmup_weight(effective['meanflow_response_weight'], epoch, args.meanflow_response_warmup_epochs))
        loader = ProgressLoader(train_loader, own.stage_key, epoch, 'training')
        if args.stage == 'teacher':
            train_loss = keyword_call(training.train_teacher_one_epoch, [model, loader, criterion, optimizer, device], values)
        else:
            train_loss = keyword_call(training.train_student_one_epoch, [teacher, model, loader, criterion, optimizer, device], values)
        assert np.isfinite(train_loss)
        if own.preflight:
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            assert gradients and all((torch.isfinite(g).all().item() for g in gradients))
            report = {'status': 'pass', 'stage': own.stage_key, 'batch_size': args.batch_size, 'finite_loss': float(train_loss), 'gradient_tensors': len(gradients), 'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(0), 'checkpoint_written': False, 'formal_training_state_reused': False, 'hidden_input_checks': hidden_checks}
            write(RUN / 'audit' / ('preflight_' + own.stage_key + '.json'), report)
            print(json.dumps(report), flush=True)
            return
        metrics = validate(epoch)
        update_best(epoch, metrics)
        scheduler.step()
        record = {'epoch': epoch, 'train_loss': float(train_loss), **metrics, 'best_auc': best_value, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'epoch_seconds': time.monotonic() - started}
        history.append(record)
        write(save_dir / 'history.json', history)
        save_last(epoch)
        write(RUN / 'worker_status.json', {'stage': own.stage_key, 'phase': 'epoch_complete', **record, 'epoch_one_based': epoch + 1, 'pid': os.getpid()})
        print(json.dumps(record), flush=True)
        if bad_epochs >= args.early_stop_patience > 0:
            break
    assert best_path.exists() and last_path.exists()
    write(save_dir / 'training_complete.json', {'stage': own.stage_key, 'best_auc': best_value, 'best_epoch': best_epoch, 'epochs_trained': sum((r['epoch'] >= 0 for r in history)), 'best_checkpoint_sha256': sha(best_path), 'protocol_sha256': expected_hash, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'resumed': resumed})
    print('Completed', own.stage_key, 'best validation AUC', best_value, flush=True)
if __name__ == '__main__':
    main()
