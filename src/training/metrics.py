import torch
import numpy as np
from collections import defaultdict

# COMPUTE EMBEDDINGS
def compute_embeddings(model, dataloader, device):
    embeddings = []
    labels = []
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            emb = model(img)
            embeddings.append(emb.cpu())
            labels.extend(label.numpy())

    embeddings = torch.cat(embeddings)
    return embeddings, np.array(labels)

# BUILD GALLERY - Tính mean các embedding cùng label để tạo 1 vector đại diện
def build_gallery(embeddings, labels):
    gallery = {}
    label_dict = defaultdict(list)

    for emb, label in zip(embeddings, labels):
        label_dict[label].append(emb)

    for label in label_dict:
        gallery[label] = torch.stack(label_dict[label]).mean(0)

    return gallery

# CREATE PAIRS - Với mỗi ảnh probe ta so sánh với all class trong gallery
# def create_pairs(probe_embs, probe_labels, gallery):
#     similarities  = []
#     labels = []
#
#     for emb, label in zip(probe_embs, probe_labels):
#         for g_label in gallery:
#             sim = torch.dot(emb, gallery[g_label]).item()
#             similarities.append(sim)
#             labels.append(1 if label == g_label else 0)
#
#     return np.array(similarities), np.array(labels)

def create_pairs(probe_embs, probe_labels, gallery):
    # stack gallery
    gallery_labels = list(gallery.keys())
    gallery_mat = torch.stack([gallery[k] for k in gallery_labels])  # [C, D]

    similarities = []
    labels = []

    for emb, label in zip(probe_embs, probe_labels):
        # cosine similarity vectorized
        sim_vec = torch.mv(gallery_mat, emb)  # [C]

        similarities.append(sim_vec)
        labels.extend([1 if label == g else 0 for g in gallery_labels])

    return np.concatenate(similarities), np.array(labels)

# FAR - False Accept Rate: nhận nhầm người lạ là đúng
# FRR - False Reject Rate: từ chối đúng người
# def compute_far_frr(similarities, labels, threshold):
#     preds = similarities > threshold
#
#     TP = np.sum((preds == 1) & (labels == 1)) # đúng là cùng người
#     TN = np.sum((preds == 0) & (labels == 0)) # đúng là khác người
#     FP = np.sum((preds == 1) & (labels == 0)) # tưởng cùng người nhưng thực ra khác
#     FN = np.sum((preds == 0) & (labels == 1)) # cùng người nhưng bị reject
#
#     FAR = FP / (FP + TN + 1e-8) # trong tất cả negative pairs, có bao nhiêu bị nhận nhầm?
#     FRR = FN / (FN + TP + 1e-8) # trong all positive pairs, có bao nhiêu bị từ chối?
#
#     return FAR, FRR, TP, TN, FP, FN
def compute_far_frr(similarities, labels, threshold):
    preds = similarities > threshold

    TP = np.sum((preds & (labels == 1)))
    TN = np.sum((~preds & (labels == 0)))
    FP = np.sum((preds & (labels == 0)))
    FN = np.sum((~preds & (labels == 1)))

    FAR = FP / (FP + TN + 1e-8)
    FRR = FN / (FN + TP + 1e-8)

    return FAR, FRR, TP, TN, FP, FN

# EER - Equal Error Rate: điểm mà FAR = FRR tức là mức cân bằng giữa 2 lỗi FAR và FRR
# def compute_eer(similarities, labels):
#     thresholds = np.linspace(similarities.min(), similarities.max(), 1000)
#
#     fars = []
#     frrs = []
#
#     for t in thresholds:
#         FAR, FRR, *_ = compute_far_frr(similarities, labels, t)
#
#         fars.append(FAR)
#         frrs.append(FRR)
#
#     fars = np.array(fars)
#     frrs = np.array(frrs)
#
#     idx = np.argmin(np.abs(fars - frrs))
#
#     eer = (fars[idx] + frrs[idx]) / 2
#     threshold = thresholds[idx]
#
#     return eer, threshold
def compute_eer(similarities, labels):
    similarities = np.array(similarities)
    labels = np.array(labels)

    # sort by score
    idx = np.argsort(similarities)
    sims = similarities[idx]
    labs = labels[idx]

    # cumulative counts
    P = np.sum(labs == 1)
    N = np.sum(labs == 0)

    TP = P
    FP = N

    best_diff = 1e9
    eer = 0
    threshold = 0

    i = 0

    for t in sims:
        while i < len(sims) and sims[i] <= t:
            if labs[i] == 1:
                TP -= 1
            else:
                FP -= 1
            i += 1

        FN = P - TP
        TN = N - FP

        FAR = FP / (FP + TN + 1e-8)
        FRR = FN / (FN + TP + 1e-8)

        diff = abs(FAR - FRR)

        if diff < best_diff:
            best_diff = diff
            eer = (FAR + FRR) / 2
            threshold = t

    return eer, threshold

# ACCURACY
def compute_accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN + 1e-8)

# TPR @ FAR: tìm TPR cao nhất nhưng vẫn đảm bảo FAR ≤ target
# FAR - False Accept Rate: nhận nhầm người lạ là đúng
def compute_tpr_at_far(similarities, labels, target_far=1e-3):
    thresholds = np.linspace(similarities.min(), similarities.max(), 2000)

    best_tpr = 0

    for t in thresholds:
        FAR, FRR, TP, TN, FP, FN = compute_far_frr(similarities, labels, t)

        if FAR <= target_far:
            TPR = TP / (TP + FN + 1e-8)
            best_tpr = max(best_tpr, TPR)

    return best_tpr

# ROC curve
# FAR - False Accept Rate: nhận nhầm người lạ là đúng
# TPR - True Positive Rate: nhận đúng người thật
# def compute_roc(similarities, labels):
#     thresholds = np.linspace(similarities.min(), similarities.max(), 1000)
#
#     tprs = []
#     fars = []
#
#     for t in thresholds:
#         FAR, FRR, TP, TN, FP, FN = compute_far_frr(similarities, labels, t)
#
#         TPR = TP / (TP + FN + 1e-8)   # recall
#         FAR = FP / (FP + TN + 1e-8)
#
#         tprs.append(TPR)
#         fars.append(FAR)
#
#     return np.array(fars), np.array(tprs), thresholds
def compute_roc(similarities, labels):
    similarities = np.array(similarities)
    labels = np.array(labels)

    # sort descending
    idx = np.argsort(-similarities)
    sims = similarities[idx]
    labs = labels[idx]

    P = np.sum(labs == 1)
    N = np.sum(labs == 0)

    TP = 0
    FP = 0

    tprs = []
    fars = []

    prev_score = None

    for i in range(len(sims)):
        if labs[i] == 1:
            TP += 1
        else:
            FP += 1

        TPR = TP / (P + 1e-8)
        FAR = FP / (N + 1e-8)

        if sims[i] != prev_score:
            tprs.append(TPR)
            fars.append(FAR)
            prev_score = sims[i]

    return np.array(fars), np.array(tprs), sims