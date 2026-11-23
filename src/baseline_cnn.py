import argparse
import gc
import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
from monai.networks.nets import DenseNet121, resnet18
from monai.utils import set_determinism
from common import RUN, SEED, align, metrics, now, rows, select_threshold, sha, validate, write
from data import get_node_dataloader
from losses import AsymmetricBinaryLoss
from robustness import student_mri_style

def save(path, value):
    temp = str(path) + '.tmp'
    torch.save(value, temp)
    os.replace(temp, str(path))

def build_model(arch):
    if arch == 'resnet18':
        return resnet18(spatial_dims=3, n_input_channels=3, num_classes=1, pretrained=False)
    return DenseNet121(spatial_dims=3, in_channels=3, out_channels=1, pretrained=False)

def loader(split, train=False, preflight=False):
    path = RUN / 'audit/preflight_train.json' if preflight else RUN / 'manifests' / (split + '.json')
    return get_node_dataloader(str(path), batch_size=12, is_train=train, use_ddp=False, stage=2, loader_workers=0 if preflight else 2, cache_workers=1, cache_rate=0.1 if train and (not preflight) else 0.0, persistent_workers=False, disable_random_aug=not train, balanced_sampler=False)[0]

def inputs(batch, device, train=False):
    assert torch.count_nonzero(batch['PET']).item() == 0
    assert torch.count_nonzero(batch['T1CE']).item() == 0
    assert torch.count_nonzero(batch['PET_present']).item() == 0
    assert torch.count_nonzero(batch['T1CE_present']).item() == 0
    x = torch.cat([batch['T1'], batch['T2'], batch['CT']], dim=1).to(device)
    y = batch['label'].view(-1, 1).to(device)
    assert x.shape[1:] == (3, 64, 64, 64) and torch.isfinite(x).all().item()
    if train:
        x = torch.cat([student_mri_style(x[:, :2]), x[:, 2:3]], dim=1)
    return (x, y)

@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    records = []
    for batch in batches:
        x, y = inputs(batch, device)
        probability = model(x).sigmoid().view(-1).cpu().numpy()
        for i, prob in enumerate(probability):
            records.append({'id': str(batch['id'][i]), 'patient_id': str(batch['patient_id'][i]), 'label': int(y[i].item()), 'prob': float(prob)})
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['resnet18', 'densenet121'], required=True)
    parser.add_argument('--mode', choices=['preflight', 'train', 'test'], required=True)
    args = parser.parse_args()
    if args.mode == 'preflight':
        protocol, digest = (None, None)
    else:
        protocol, digest = validate()
    set_determinism(seed=SEED)
    torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.35, 0)
    device = torch.device('cuda:0')
    model = build_model(args.arch).to(device)
    directory = RUN / 'checkpoints' / args.arch
    directory.mkdir(exist_ok=True)
    best, last = (directory / 'best.pth', directory / 'last.pth')
    if args.mode == 'test':
        assert (RUN / 'all_training_complete.json').is_file()
        locked = json.loads((directory / 'selection.json').read_text())
        assert locked['protocol_sha256'] == digest and sha(best) == locked['checkpoint_sha256']
        model.load_state_dict(torch.load(str(best), map_location=device), strict=True)
        frame = align(predict(model, loader('test'), device), rows('test'))
        threshold = locked['threshold']
        frame['threshold'] = threshold
        frame['pred_threshold'] = (frame.prob >= threshold).astype(int)
        frame.to_csv(RUN / 'predictions' / (args.arch + '_test.csv'), index=False)
        write(RUN / 'metrics' / (args.arch + '_test.json'), metrics(frame.label, frame.prob, threshold))
        return
    train_loader = loader('train', train=True, preflight=args.mode == 'preflight')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = AsymmetricBinaryLoss(gamma_pos=1.0, gamma_neg=4.0, clip=0.05)
    if args.mode == 'preflight':
        model.train()
        batch = next(iter(train_loader))
        x, y = inputs(batch, device, train=True)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        assert torch.isfinite(loss).item()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all((torch.isfinite(g).all().item() for g in gradients))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            eval_output = model(inputs(batch, device)[0])
        assert eval_output.shape == (12, 1) and torch.isfinite(eval_output).all().item()
        report = {'state': 'pass', 'architecture': args.arch, 'batch_size': 12, 'input_shape': list(x.shape), 'loss': float(loss.item()), 'parameters': sum((p.numel() for p in model.parameters())), 'finite_gradient_tensors': len(gradients), 'max_cuda_allocated_bytes': torch.cuda.max_memory_allocated(0), 'PET_and_T1CE_paths_absent': True, 'PET_and_T1CE_pixels_and_presence_zero': True, 'raw_CNN_input_channels': ['T1', 'T2', 'CT'], 'clinical_metadata_forwarded': False, 'training_patients_only': True, 'preflight_state_reused_in_formal_training': False, 'formal_checkpoint_written': False}
        write(RUN / 'audit' / ('preflight_' + args.arch + '.json'), report)
        print(json.dumps(report), flush=True)
        return
    val_loader = loader('val')
    start_epoch, best_auc, bad_epochs, history = (0, -1.0, 0, [])
    if last.exists():
        state = torch.load(str(last), map_location=device)
        assert state['protocol_sha256'] == digest and state['architecture'] == args.arch
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, best_auc, bad_epochs, history = (state['epoch'] + 1, state['best_auc'], state['bad_epochs'], state['history'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), 0)
        del state
    write(directory / 'config.json', {'protocol_sha256': digest, 'arch': args.arch, 'seed': SEED, 'parameters': sum((p.numel() for p in model.parameters())), 'pretrained': False, 'inputs': ['T1', 'T2', 'CT'], 'metadata_input': False, 'optimizer': 'AdamW', 'lr': 0.0002, 'weight_decay': 0.0001, 'batch_size': 12, 'max_epochs': 50, 'patience': 5, 'loss': {'name': 'ASL', 'gamma_pos': 1.0, 'gamma_neg': 4.0, 'clip': 0.05}, 'random_augmentation': 'Same sealed V9 physical data transforms and student MRI intensity style augmentation'})
    if bad_epochs >= 5:
        start_epoch = 50
    for epoch in range(start_epoch, 50):
        started = time.monotonic()
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_loader, 1):
            x, y = inputs(batch, device, train=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            assert torch.isfinite(loss).item()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            if i == 1 or i % 20 == 0 or i == len(train_loader):
                write(RUN / (args.arch + '_status.json'), {'state': 'training', 'epoch': epoch + 1, 'batch': i, 'batches': len(train_loader), 'elapsed_seconds': time.monotonic() - started, 'last_loss': float(loss.item()), 'pid': os.getpid(), 'updated_utc': now()})
        frame = align(predict(model, val_loader, device), rows('val'))
        threshold = select_threshold(frame.label.to_numpy(), frame.prob.to_numpy())
        result = metrics(frame.label, frame.prob, threshold)
        improved = result['auc'] > best_auc + 1e-12
        if improved:
            best_auc, bad_epochs = (result['auc'], 0)
            save(best, model.state_dict())
            frame.to_csv(RUN / 'predictions' / (args.arch + '_validation_best.csv'), index=False)
            write(directory / 'selection.json', {'protocol_sha256': digest, 'best_epoch': epoch + 1, 'validation': result, 'threshold': threshold, 'checkpoint_sha256': sha(best), 'test_used_for_selection': False})
        else:
            bad_epochs += 1
        scheduler.step()
        record = {'epoch': epoch + 1, 'train_loss': total_loss / len(train_loader), 'validation': result, 'best_auc': best_auc, 'bad_epochs': bad_epochs, 'epoch_seconds': time.monotonic() - started}
        history.append(record)
        write(directory / 'history.json', history)
        save(last, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_auc': best_auc, 'bad_epochs': bad_epochs, 'history': history, 'protocol_sha256': digest, 'architecture': args.arch, 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(0)})
        write(RUN / (args.arch + '_status.json'), {'state': 'epoch_complete', **record, 'pid': os.getpid(), 'updated_utc': now()})
        print(json.dumps(record), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        if bad_epochs >= 5:
            break
    assert best.is_file() and last.is_file()
    write(directory / 'training_complete.json', {'architecture': args.arch, 'protocol_sha256': digest, 'epochs': len(history), 'best_auc': best_auc, 'checkpoint_sha256': sha(best), 'finished_utc': now()})
    write(RUN / (args.arch + '_status.json'), {'state': 'training_complete', 'epochs': len(history), 'best_auc': best_auc, 'finished_utc': now()})
if __name__ == '__main__':
    main()
