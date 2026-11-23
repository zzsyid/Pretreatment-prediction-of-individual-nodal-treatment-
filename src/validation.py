import numpy as np
import torch
from tqdm import tqdm
from objectives import extract_batch, apply_meta_ablation, apply_input_mode, make_student_present_dict, apply_student_input_mode, mask_images_by_present

def get_best_threshold_strict(y_true, y_probs):
    from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    if len(thresholds) > 0:
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        best_index = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_index])
    else:
        best_threshold = 0.5
    y_pred = (y_probs >= best_threshold).astype(int)
    return {'best_threshold': float(best_threshold), 'f1_score': round(float(f1_score(y_true, y_pred, zero_division=0)), 4), 'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4), 'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4)}

@torch.no_grad()
def validate_strict(model, dataloader, criterion, device, stage, input_mode, disable_meta, student_hide_prob_t1c=1.0, student_hide_prob_pet=1.0):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []
    pbar = tqdm(dataloader, desc='Strict validating', leave=False)
    for batch in pbar:
        mri_imgs, ct_pet_imgs, mri_present, ct_present, vols, regions, labels = extract_batch(batch, device)
        vols, regions = apply_meta_ablation(vols, regions, disable_meta)
        mri_imgs, mri_present, ct_pet_imgs, ct_present = apply_input_mode(mri_imgs, mri_present, ct_pet_imgs, ct_present, input_mode)
        if stage == 'student':
            mri_present, ct_present = make_student_present_dict(mri_present, ct_present, hide_prob_t1c=student_hide_prob_t1c, hide_prob_pet=student_hide_prob_pet)
            mri_present, ct_present = apply_student_input_mode(mri_present, ct_present, input_mode)
            mri_imgs, ct_pet_imgs = mask_images_by_present(mri_imgs, ct_pet_imgs, mri_present, ct_present)
        logits, _, _ = model(mri_imgs, mri_present, ct_pet_imgs, ct_present, vols, regions)
        loss = criterion(logits, labels)
        val_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        all_preds.extend(preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    from sklearn.metrics import accuracy_score, roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    performance = get_best_threshold_strict(all_labels, all_probs)
    return {'loss': val_loss / len(dataloader), 'auc': float(auc), 'f1': performance['f1_score'], 'acc': float(accuracy_score(all_labels, all_preds)), 'sensitivity': performance['recall'], 'threshold': performance['best_threshold'], 'precision': performance['precision']}
