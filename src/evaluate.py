import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ===================== 路径配置（和仓库目录完全匹配，无需修改） =====================
PRED_ROOT = "outputs/predictions/ablation"
METRIC_OUT = "outputs/metrics/ablation_metrics.csv"
FIG_SAVE_DIR = "outputs/figures"
os.makedirs(FIG_SAVE_DIR, exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

# 二分类标签固定配置
POS_LABEL = "risk"
LABEL_LIST = ["normal", "risk"]


def calc_all_metrics(df: pd.DataFrame, model_name: str):
    """输入单模型预测csv，返回全套评估指标"""
    y_true_raw = df["true_label"]
    y_pred_raw = df["pred_label"]
    y_score = df["risk_prob"]
    # 标签转0/1
    y_true_bin = (y_true_raw == POS_LABEL).astype(int)

    acc = accuracy_score(y_true_raw, y_pred_raw)
    prec = precision_score(y_true_raw, y_pred_raw, pos_label=POS_LABEL)
    rec = recall_score(y_true_raw, y_pred_raw, pos_label=POS_LABEL)
    f1 = f1_score(y_true_raw, y_pred_raw, pos_label=POS_LABEL)
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    auc_val = auc(fpr, tpr)
    return {
        "model_group": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc_val, 4),
        "fpr": fpr,
        "tpr": tpr
    }


def save_confusion_heatmap(df: pd.DataFrame, model_name: str):
    """生成单模型混淆矩阵图并保存"""
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=LABEL_LIST)
    plt.figure(figsize=(5, 4), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["正常样本", "风险样本"],
                yticklabels=["正常样本", "风险样本"])
    plt.title(f"{model_name} 混淆矩阵")
    plt.xlabel("模型预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    save_path = os.path.join(FIG_SAVE_DIR, f"cm_{model_name}.png")
    plt.savefig(save_path)
    plt.close()


def draw_roc_compare(metric_list, save_path):
    """多张模型ROC曲线合并一张对比图"""
    plt.figure(figsize=(7, 6), dpi=300)
    for item in metric_list:
        plt.plot(item["fpr"], item["tpr"], lw=2,
                 label=f'{item["model_group"]} (AUC={item["roc_auc"]:.3f})')
    # 随机猜测基准线
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1.03)
    plt.xlabel("假阳性率 FPR")
    plt.ylabel("真阳性率 TPR")
    plt.title("各消融模型 ROC 曲线对比")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def draw_ablation_f1_bar(metric_df, save_path):
    """消融实验F1柱状对比图"""
    plt.figure(figsize=(9, 5), dpi=300)
    sns.barplot(data=metric_df, x="model_group", y="f1", palette="viridis")
    plt.title("不同特征组合消融实验 F1 值对比")
    plt.xlabel("特征组合方案")
    plt.ylabel("F1 分数")
    plt.ylim(0.5, 0.95)
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
# 1. 批量读取ablation下所有预测csv
pred_files = sorted([f for f in os.listdir(PRED_ROOT) if f.endswith("_pred.csv")])
    all_model_metrics = []
    roc_draw_cache = []

    print("===== 开始批量读取消融预测文件 =====")
    for file_name in pred_files:
file_path = os.path.join(PRED_ROOT, file_name)
model_name = file_name[: -len("_pred.csv")]
df = pd.read_csv(file_path, encoding="gbk")
        print(f"正在计算：{model_name}")
        # 计算指标
        metric_dict = calc_all_metrics(df, model_name)
        all_model_metrics.append(metric_dict)
        roc_draw_cache.append(metric_dict)
        # 保存混淆矩阵
        save_confusion_heatmap(df, model_name)

    # 2. 汇总指标表格，写入模板csv
    metric_df = pd.DataFrame(all_model_metrics)
    # 剔除绘图用的fpr/tpr列，只保留数值指标
export_df = metric_df.drop(columns=["fpr", "tpr"])
export_df.to_csv(METRIC_OUT, index=False, encoding="utf-8")
print(f"\n✅ 全部指标汇总完成，保存至：{METRIC_OUT}")

    # 3. 绘制两张总对比图
    draw_roc_compare(roc_draw_cache, os.path.join(FIG_SAVE_DIR, "roc_all_compare.png"))
    draw_ablation_f1_bar(metric_df, os.path.join(FIG_SAVE_DIR, "ablation_f1_bar.png"))
    print(f"✅ 全部图表已输出至：{FIG_SAVE_DIR}")
    print("\n===== 评测流程全部结束 =====")
print("产出清单：")
print("1. 指标汇总表：outputs/metrics/ablation_metrics.csv")
print("2. 各模型混淆矩阵：outputs/figures/cm_*.png")
print("3. ROC对比图：outputs/figures/roc_all_compare.png")
print("4. 消融F1柱状图：outputs/figures/ablation_f1_bar.png")


if __name__ == "__main__":
    main()