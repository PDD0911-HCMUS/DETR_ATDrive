import re
import os
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_log(log_path):
    """
    Parse a DETR-style training log to extract:
    - Per-epoch training stats (loss, loss_ce, loss_bbox, loss_giou, loss_mask, loss_dice, class_error)
    - Per-eval AP (0.50:0.95 all) and AR (0.50:0.95 all, maxDets=100)
    """

    # Lưu theo epoch
    train_stats = defaultdict(list)
    eval_stats = defaultdict(list)

    # Regex cho dòng train (Epoch: [xx] ...)
    # Ví dụ:
    # Epoch: [0]  [12200/33403]  ... loss: 23.9385 (...)  loss_ce: 1.01 (...) loss_bbox: ...
    epoch_re = re.compile(r"Epoch:\s*\[(\d+)\]")
    float_re = re.compile(r"([a-zA-Z_]+):\s*([\d\.]+)")

    # Regex cho AP/AR từ COCO eval
    ap_all_re = re.compile(
        r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0.50:0.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)"
    )
    ar_all_re = re.compile(
        r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0.50:0.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)"
    )

    current_epoch = None
    eval_pending_for_epoch = None  # epoch tương ứng với eval gần nhất

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            # 1) Bắt epoch ở dòng train
            m_ep = epoch_re.search(line)
            if m_ep:
                current_epoch = int(m_ep.group(1))
                # Tạm gắn eval của epoch này nếu có evaluation sau đó
                eval_pending_for_epoch = current_epoch

                # Parse các loss trong cùng dòng
                # Ví dụ: "loss: 23.9385 (25.3921)  loss_ce: 1.0119 (1.2003) ..."
                for m in float_re.finditer(line):
                    key = m.group(1)
                    val = float(m.group(2))
                    # Lọc các key chính cho gọn
                    if key in [
                        "loss",
                        "loss_ce",
                        "loss_bbox",
                        "loss_giou",
                        "loss_mask",
                        "loss_dice",
                        "class_error",
                    ]:
                        train_stats[key].append((current_epoch, val))

            # 2) Parse AP @ 0.50:0.95 (all, maxDets=100)
            m_ap = ap_all_re.search(line)
            if m_ap and eval_pending_for_epoch is not None:
                val = float(m_ap.group(1))
                eval_stats["AP_0.5_0.95_all"].append((eval_pending_for_epoch, val))

            # 3) Parse AR @ 0.50:0.95 (all, maxDets=100)
            m_ar = ar_all_re.search(line)
            if m_ar and eval_pending_for_epoch is not None:
                val = float(m_ar.group(1))
                eval_stats["AR_0.5_0.95_all"].append((eval_pending_for_epoch, val))

    return train_stats, eval_stats


def plot_curves(train_stats, eval_stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Helper nhỏ
    def _plot_single(metric_name, stats_dict, ylabel, filename):
        if metric_name not in stats_dict or len(stats_dict[metric_name]) == 0:
            print(f"[WARN] No data for {metric_name}, skip plot {filename}")
            return
        epochs, vals = zip(*stats_dict[metric_name])
        plt.figure()
        plt.plot(epochs, vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(metric_name)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

    # 1) Loss tổng
    _plot_single("loss", train_stats, "Loss", "train_loss.png")

    # 2) Các loss thành phần
    for m in ["loss_ce", "loss_bbox", "loss_giou", "loss_mask", "loss_dice"]:
        _plot_single(m, train_stats, m, f"{m}.png")

    # 3) class_error
    _plot_single("class_error", train_stats, "Class Error (%)", "class_error.png")

    # 4) AP & AR trên cùng 1 hình
    if (
        "AP_0.5_0.95_all" in eval_stats
        and len(eval_stats["AP_0.5_0.95_all"]) > 0
        and "AR_0.5_0.95_all" in eval_stats
        and len(eval_stats["AR_0.5_0.95_all"]) > 0
    ):
        ep_ap, val_ap = zip(*eval_stats["AP_0.5_0.95_all"])
        ep_ar, val_ar = zip(*eval_stats["AR_0.5_0.95_all"])

        plt.figure()
        plt.plot(ep_ap, val_ap, marker="o", label="AP[0.50:0.95] all")
        plt.plot(ep_ar, val_ar, marker="s", label="AR[0.50:0.95] all")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("COCO AP/AR vs Epoch")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ap_ar.png"))
        plt.close()
    else:
        print("[WARN] Not enough AP/AR data to plot ap_ar.png")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from DETR-style log.")
    parser.add_argument("--log", type=str, required=True, help="Path to log file (.log)")
    parser.add_argument("--out", type=str, default="plots", help="Output directory for plots")
    args = parser.parse_args()

    print(f"[INFO] Parsing log: {args.log}")
    train_stats, eval_stats = parse_log(args.log)

    print("[INFO] Found metrics:")
    for k in train_stats:
        print(f"  Train {k}: {len(train_stats[k])} points")
    for k in eval_stats:
        print(f"  Eval {k}: {len(eval_stats[k])} points")

    print(f"[INFO] Saving plots to: {args.out}")
    plot_curves(train_stats, eval_stats, args.out)
    print("[DONE] All plots generated.")


if __name__ == "__main__":
    main()
