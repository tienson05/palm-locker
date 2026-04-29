import json
import os
from datetime import datetime

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from metrics import *
from src.datasets.eval_dataset import EvalDataset
from src.model.palm_net import PalmNet
from src.transforms.transform_pipeline import eval_transform

exp_name = "palmnet_v1"
run_id = datetime.now().strftime(f"{exp_name}__%Y-%m-%d__%H-%M-%S")
save_dir = os.path.join("../../results", run_id)
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
gallery_dataset = EvalDataset("../../data/splits/test/session1", eval_transform)
probe_dataset = EvalDataset("../data/splits/test/session2", eval_transform)

gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False) # final batch = 48 imgs (1200 imgs)
probe_loader = DataLoader(probe_dataset, batch_size=64, shuffle=False) # final batch = 48 imgs (1200 imgs)

# Model
model = PalmNet().to(device)
model.load_state_dict(torch.load("../../models/palmnet_v1.pth", map_location=device))
model.eval()

# COMPUTE EMBEDDINGS: gallery_emb = (1200, 128); gallery_labels = (1200,)
gallery_emb, gallery_labels = compute_embeddings(model, gallery_loader, device)
gallery = build_gallery(gallery_emb, gallery_labels) # gallery dict chứa vector đại diện cho 1 label

probe_emb, probe_labels = compute_embeddings(model, probe_loader, device)

# distances = pair_labels = (1200 * 600) phần tử
similarities, pair_labels = create_pairs(probe_emb, probe_labels, gallery) # CREATE PAIRS

pos = similarities[pair_labels == 1] # lấy giá trị khoảng cách tại các vị trí cùng class
neg = similarities[pair_labels == 0] # lấy giá trị khoảng cách tại các vị trí khác class
# print(f"pos min: {pos.min()} | pos max: {pos.max()} | pos mean: {pos.mean()}") # tính mean, mong muốn là cao vì cùng class và dùng cosime similarity
# print(f"neg min: {pos.min()} | neg max: {neg.max()} | neg mean: {neg.mean()}") # tính mean, mong muốn là thấp vì khác class và dùng cosime similarity
score_stats = {
    "pos_mean": float(pos.mean()),
    "pos_std": float(pos.std()),
    "pos_min": float(pos.min()),
    "pos_max": float(pos.max()),

    "neg_mean": float(neg.mean()),
    "neg_std": float(neg.std()),
    "neg_min": float(neg.min()),
    "neg_max": float(neg.max()),

    "separation_gap": float(pos.mean() - neg.mean())
}
print("\n===== SCORE STATS =====")
print(score_stats)

# METRICS
eer, threshold = compute_eer(similarities, pair_labels)
FAR, FRR, TP, TN, FP, FN = compute_far_frr(similarities, pair_labels, threshold)
acc = compute_accuracy(TP, TN, FP, FN)
tpr_at_far  = compute_tpr_at_far(similarities, pair_labels, target_far=1e-3)

# ROC
fars, tprs, thresholds = compute_roc(similarities, pair_labels)
plt.figure(figsize=(6, 6))
plt.plot(fars, tprs, label=f"EER={eer:.4f}")
plt.plot([0, 1], [0, 1], "--", label="random")
plt.xlabel("False Accept Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.grid()
plt.legend()
plt.savefig(os.path.join(save_dir, "roc.png"), dpi=300, bbox_inches="tight")
plt.close()

# METRICS JSON
results = {
    "experiment": exp_name,
    "eer": float(eer),
    "threshold": float(threshold),
    "far": float(FAR),
    "frr": float(FRR),
    "accuracy": float(acc),
    "tpr_at_far_1e-3": float(tpr_at_far),
    "confusion_matrix": {
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN)
    },
    "score_statistics": score_stats
}
with open(os.path.join(save_dir, "metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

# HUMAN SUMMARY
summary_path = os.path.join(save_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write("===== PALMNET EVALUATION =====\n")
    f.write(f"EER: {eer:.6f}\n")
    f.write(f"Threshold: {threshold:.6f}\n")
    f.write(f"FAR: {FAR:.6f}\n")
    f.write(f"FRR: {FRR:.6f}\n")
    f.write(f"Accuracy: {acc:.6f}\n")
    f.write(f"TPR@FAR=1e-3: {tpr_at_far:.6f}\n\n")

    f.write("===== SCORE GAP =====\n")
    f.write(f"Pos mean: {pos.mean():.6f}\n")
    f.write(f"Neg mean: {neg.mean():.6f}\n")
    f.write(f"Separation: {pos.mean() - neg.mean():.6f}\n")

# PRINT FINAL
print("\n==============================")
print("      TEST RESULTS")
print("==============================")
print(f"EER              : {eer:.6f}")
print(f"Threshold        : {threshold:.6f}")
print(f"FAR              : {FAR:.6f}")
print(f"FRR              : {FRR:.6f}")
print(f"Accuracy         : {acc:.6f}")
print(f"TPR@FAR=1e-3     : {tpr_at_far:.6f}")
print("\nSaved to:", save_dir)