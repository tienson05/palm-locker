from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import math

# =====================================================
# CONFIG
# =====================================================
EVENT_FILE = r"D:\Projects\Personal\PalmLocker\runs\arcface_2026-05-26_22-04-28\events.out.tfevents.1779833068.8b16c1a3528d.57.0"

SMOOTH_WEIGHT = 0.6

GROUPS = [
    "Loss",
    "Metric",
    "Distance",
    "Confusion"
]

# =====================================================
# LOAD EVENT FILE
# =====================================================
ea = event_accumulator.EventAccumulator(EVENT_FILE)
ea.Reload()

print("\nAll scalar tags:")
for tag in ea.Tags()["scalars"]:
    print(" -", tag)

# =====================================================
# TITLE FORMAT
# =====================================================
ABBREVIATIONS = {
    "TPR",
    "FPR",
    "FAR",
    "FRR",
    "EER",
    "AUC",
    "ROC",
    "F1",
    "AP",
    "TAR", "FN", "FP", "TN", "TP"
}


def pretty_title(tag):

    name = tag.split("/")[-1]

    parts = name.split("_")
    formatted = []

    for p in parts:

        if p.upper() in ABBREVIATIONS:
            formatted.append(p.upper())

        elif p.lower() == "at":
            formatted.append("@")

        else:
            formatted.append(p.capitalize())

    return " ".join(formatted)


# =====================================================
# SMOOTHING (TensorBoard-like EMA)
# =====================================================
def smooth(values, weight=0.6):

    if len(values) == 0:
        return values

    smoothed = []
    last = values[0]

    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)

    return smoothed


# =====================================================
# LAYOUT
# =====================================================
def get_layout(n):

    if n == 1:
        return 1, 1

    if n == 2:
        return 1, 2

    if n == 3:
        return 1, 3

    if n == 4:
        return 2, 2

    if n <= 6:
        return 2, 3

    return math.ceil(n / 3), 3


# =====================================================
# SAVE ONE GROUP
# =====================================================
def save_group_figure(group_name):

    tags = [
        tag for tag in ea.Tags()["scalars"]
        if tag.startswith(f"{group_name}/")
    ]

    # loại bỏ các tag không muốn hiển thị
    tags = [
        tag for tag in tags
        if not tag.endswith("_step")
    ]

    tags.sort()

    if len(tags) == 0:
        print(f"Skip {group_name}: no tags found")
        return

    print(f"\n[{group_name}]")

    for tag in tags:
        print(" -", tag)

    n = len(tags)

    rows, cols = get_layout(n)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 3.5 * rows)
    )

    # Chuẩn hóa axes
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # =================================================
    # PLOT
    # =================================================
    for idx, tag in enumerate(tags):

        events = ea.Scalars(tag)

        steps = [e.step for e in events]
        values = [e.value for e in events]

        smooth_values = smooth(
            values,
            SMOOTH_WEIGHT
        )

        ax = axes[idx]

        # Raw curve
        ax.plot(
            steps,
            values,
            linewidth=1,
            alpha=0.25
        )

        # Smoothed curve
        ax.plot(
            steps,
            smooth_values,
            linewidth=2.2
        )

        ax.set_title(pretty_title(tag))

        ax.set_xlabel("Epoch")

        ax.grid(
            True,
            alpha=0.3
        )

        ax.set_facecolor("#fafafa")

    # =================================================
    # HIDE UNUSED SUBPLOTS
    # =================================================
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    pdf_name = f"{group_name.lower()}_report.pdf"
    png_name = f"{group_name.lower()}_report.png"

    plt.savefig(
        pdf_name,
        bbox_inches="tight"
    )

    plt.savefig(
        png_name,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {pdf_name}")
    print(f"Saved: {png_name}")


# =====================================================
# GLOBAL STYLE
# =====================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10
})

# =====================================================
# EXPORT ALL FIGURES
# =====================================================
for group in GROUPS:
    save_group_figure(group)

print("\nDone.")