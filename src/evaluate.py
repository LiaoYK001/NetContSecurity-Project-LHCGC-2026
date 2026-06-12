"""成员 C：统一评估消融预测并生成图表。

Member C: evaluate ablation predictions and generate figures.

中文：本脚本只使用 ``split=test`` 样本计算最终指标，并输出指标表、混淆矩阵、ROC 曲线和 F1 柱状图。
English: This script computes final metrics only on ``split=test`` samples and exports the metrics table, confusion matrices, ROC curves, and F1 bar chart.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

# 路径配置：预测 CSV 由 src/run_ablation.py 生成，默认不提交 Git。
DATASET_PATH = Path("data/processed/dataset_v1.csv")
PRED_ROOT = Path("outputs/predictions/ablation")
METRIC_OUT = Path("outputs/metrics/ablation_metrics.csv")
FIG_SAVE_DIR = Path("outputs/figures")

POS_LABEL = "risk"
LABEL_LIST = ["normal", "risk"]
REQUIRED_COLUMNS = {"sample_id", "true_label", "pred_label", "risk_prob", "model_name"}
REQUIRED_DATASET_COLUMNS = {"sample_id", "split"}


def read_prediction_csv(path: Path) -> pd.DataFrame:
    """读取预测 CSV，兼容项目中常见的 UTF-8/GBK 编码。

    Read a prediction CSV and support common UTF-8/GBK encodings used in this project.
    """
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def validate_prediction_frame(df: pd.DataFrame, path: Path) -> None:
    """校验预测表是否满足统一 5 列接口和标签要求。

    Validate that a prediction table satisfies the unified 5-column interface and label contract.
    """
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} 缺少必要字段：{missing}")
    if df.empty:
        raise ValueError(f"{path} 没有任何预测行")
    unknown_true = sorted(set(df["true_label"]) - set(LABEL_LIST))
    unknown_pred = sorted(set(df["pred_label"]) - set(LABEL_LIST))
    if unknown_true:
        raise ValueError(f"{path} true_label 含未知标签：{unknown_true}")
    if unknown_pred:
        raise ValueError(f"{path} pred_label 含未知标签：{unknown_pred}")
    if df["risk_prob"].isna().any():
        raise ValueError(f"{path} risk_prob 存在空值")


def load_test_sample_ids(dataset_path: Path) -> set[str] | None:
    """优先用主数据表的 split=test 固定最终评估口径。

    Prefer the master dataset's ``split=test`` rows to lock the final evaluation scope.
    """
    if not dataset_path.exists():
        print(f"[WARN] 未找到主数据表：{dataset_path}，将使用预测 CSV 全量行进行保底评估")
        return None

    dataset = read_prediction_csv(dataset_path)
    missing = sorted(REQUIRED_DATASET_COLUMNS - set(dataset.columns))
    if missing:
        raise ValueError(f"{dataset_path} 缺少必要字段：{missing}")

    test_rows = dataset[dataset["split"].astype(str) == "test"]
    if test_rows.empty:
        raise ValueError(f"{dataset_path} 中没有 split=test 样本，无法生成最终评估指标")

    test_ids = set(test_rows["sample_id"].astype(str))
    print(f"[INFO] 最终评估口径：仅使用 split=test 样本，数量={len(test_ids)}")
    return test_ids


def filter_to_test_split(df: pd.DataFrame, test_ids: set[str] | None, path: Path) -> pd.DataFrame:
    """把预测表过滤到最终 test 样本。

    Filter a prediction table to the final test samples.
    """
    if test_ids is None:
        return df.copy()

    sample_ids = df["sample_id"].astype(str)
    pred_ids = set(sample_ids)
    missing_ids = sorted(test_ids - pred_ids)
    if missing_ids:
        preview = missing_ids[:5]
        raise ValueError(f"{path} 缺少 split=test 预测样本 {len(missing_ids)} 个，例如：{preview}")

    filtered = df[sample_ids.isin(test_ids)].copy()
    if filtered.empty:
        raise ValueError(f"{path} 没有任何 split=test 预测行")
    return filtered


def calc_all_metrics(df: pd.DataFrame, model_name: str) -> dict[str, object]:
    """输入单模型预测 CSV，返回评估指标和 ROC 绘图数据。

    Take one model's prediction CSV and return metrics plus ROC plotting data.
    """
    y_true_raw = df["true_label"]
    y_pred_raw = df["pred_label"]
    y_score = pd.to_numeric(df["risk_prob"], errors="raise")
    y_true_bin = (y_true_raw == POS_LABEL).astype(int)

    if y_true_bin.nunique() < 2:
        fpr = [0.0, 1.0]
        tpr = [0.0, 1.0]
        roc_auc = 0.5
        print(f"[WARN] {model_name} 评估样本只有单一类别，ROC AUC 按 0.5 保底记录")
    else:
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)

    return {
        "model_group": model_name,
        "sample_count": int(len(df)),
        "accuracy": round(accuracy_score(y_true_raw, y_pred_raw), 4),
        "precision": round(
            precision_score(y_true_raw, y_pred_raw, pos_label=POS_LABEL, zero_division=0),
            4,
        ),
        "recall": round(
            recall_score(y_true_raw, y_pred_raw, pos_label=POS_LABEL, zero_division=0),
            4,
        ),
        "f1": round(
            f1_score(y_true_raw, y_pred_raw, pos_label=POS_LABEL, zero_division=0),
            4,
        ),
        "roc_auc": round(roc_auc, 4),
        "fpr": fpr,
        "tpr": tpr,
    }


def save_confusion_heatmap(df: pd.DataFrame, model_name: str) -> None:
    """保存单个模型的混淆矩阵图片。

    Save the confusion-matrix figure for one model.
    """
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=LABEL_LIST)
    plt.figure(figsize=(5, 4), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["normal", "risk"],
        yticklabels=["normal", "risk"],
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(FIG_SAVE_DIR / f"cm_{model_name}.png")
    plt.close()


def draw_roc_compare(metric_list: list[dict[str, object]], save_path: Path) -> None:
    """绘制五组消融模型的 ROC 对比图。

    Draw the ROC comparison figure for the five ablation models.
    """
    plt.figure(figsize=(7, 6), dpi=300)
    for item in metric_list:
        plt.plot(
            item["fpr"],
            item["tpr"],
            lw=2,
            label=f'{item["model_group"]} (AUC={item["roc_auc"]:.3f})',
        )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1.03)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Ablation ROC Comparison")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def draw_ablation_f1_bar(metric_df: pd.DataFrame, save_path: Path) -> None:
    """绘制五组消融模型的 F1 柱状图。

    Draw the F1 bar chart for the five ablation models.
    """
    plt.figure(figsize=(9, 5), dpi=300)
    sns.barplot(data=metric_df, x="model_group", y="f1", hue="model_group", palette="viridis", legend=False)
    plt.title("Ablation F1 Comparison")
    plt.xlabel("Feature Group")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def find_prediction_files() -> list[Path]:
    if not PRED_ROOT.exists():
        return []
    return sorted(PRED_ROOT.glob("*_pred.csv"))


def main() -> int:
    """执行统一评估、导出指标和图表。

    Run unified evaluation and export metrics plus figures.
    """
    FIG_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_OUT.parent.mkdir(parents=True, exist_ok=True)

    pred_files = find_prediction_files()
    if not pred_files:
        print(f"[ERROR] 未找到消融预测文件：{PRED_ROOT}/*_pred.csv")
        print("请先运行：uv run python src/run_ablation.py --dataset data/processed/dataset_v1.csv --mode scores")
        return 1

    test_ids = load_test_sample_ids(DATASET_PATH)
    all_model_metrics: list[dict[str, object]] = []
    print("===== 开始批量读取消融预测文件 =====")
    for file_path in pred_files:
        model_name = file_path.name[: -len("_pred.csv")]
        df = read_prediction_csv(file_path)
        validate_prediction_frame(df, file_path)
        eval_df = filter_to_test_split(df, test_ids, file_path)
        print(f"正在计算：{model_name}，评估样本数={len(eval_df)}")
        metric_dict = calc_all_metrics(eval_df, model_name)
        all_model_metrics.append(metric_dict)
        save_confusion_heatmap(eval_df, model_name)

    metric_df = pd.DataFrame(all_model_metrics)
    export_df = metric_df.drop(columns=["fpr", "tpr"])
    export_df.to_csv(METRIC_OUT, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 全部指标汇总完成，保存至：{METRIC_OUT}")

    draw_roc_compare(all_model_metrics, FIG_SAVE_DIR / "roc_all_compare.png")
    draw_ablation_f1_bar(metric_df, FIG_SAVE_DIR / "ablation_f1_bar.png")
    print(f"[OK] 全部图表已输出至：{FIG_SAVE_DIR}")
    print("\n===== 评测流程全部结束 =====")
    print("产出清单：")
    print(f"1. 指标汇总表：{METRIC_OUT}")
    print("2. 各模型混淆矩阵：outputs/figures/cm_*.png")
    print("3. ROC对比图：outputs/figures/roc_all_compare.png")
    print("4. 消融F1柱状图：outputs/figures/ablation_f1_bar.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
