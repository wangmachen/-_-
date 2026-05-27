# -*- coding: utf-8 -*-
"""
生成论文所需的所有独立图表（12张）+ 2张组合图
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve,
    auc
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
import os
output_dir = r'C:\Users\pozhuzhishi\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a16adfd98a20a2e4c1e8384'
os.chdir(output_dir)

# =============================================================================
# 1. 数据加载
# =============================================================================
print("加载数据...")
# 尝试从多个可能的位置加载数据
data_paths = [
    r'C:\Users\pozhuzhishi\Desktop\数据挖掘数据集\creditcard.csv\creditcard.csv',
    r'C:\Users\pozhuzhishi\Desktop\数据挖掘数据集\creditcard.csv',
    r'C:\Users\pozhuzhishi\.trae-cn\attachments\creditcard.csv',
    'creditcard.csv',
    os.path.join(output_dir, 'creditcard.csv'),
]
df = None
for p in data_paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"从 {p} 加载数据成功")
        break

if df is None:
    print("ERROR: 未找到 creditcard.csv 数据文件！")
    print("请将数据集文件放到以下目录之一：")
    for p in data_paths:
        print(f"  {p}")
    exit(1)

print(f"数据集维度: {df.shape}")

# =============================================================================
# 2. 数据预处理
# =============================================================================
X = df.drop(columns=['Class'])
y = df['Class']
X['Time'] = X['Time'] / 3600

scaler_amount = RobustScaler()
scaler_time = RobustScaler()
scaler_others = StandardScaler()

X['Amount'] = scaler_amount.fit_transform(X[['Amount']])
X['Time'] = scaler_time.fit_transform(X[['Time']])
other_cols = [c for c in X.columns if c not in ['Amount', 'Time']]
X[other_cols] = scaler_others.fit_transform(X[other_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# =============================================================================
# 3. 模型训练
# =============================================================================
print("训练模型...")
rf_base = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
)
rf_base.fit(X_train_smote, y_train_smote)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='f1', n_jobs=-1, verbose=0
)
rf_grid.fit(X_train_smote, y_train_smote)
rf_best = rf_grid.best_estimator_
print(f"最佳参数: {rf_grid.best_params_}")

y_pred = rf_best.predict(X_test)
y_prob = rf_best.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

fraud_count = df['Class'].value_counts()
fraud = df[df['Class'] == 1]

# =============================================================================
# 4. 生成12张独立图表
# =============================================================================
print("生成独立图表...")

# --- 图1: 类别分布 ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie([fraud_count[0], fraud_count[1]], labels=['正常交易', '欺诈交易'],
       autopct='%1.4f%%', colors=['#4CAF50', '#F44336'], explode=(0, 0.1),
       textprops={'fontsize': 12})
ax.set_title('交易类别分布', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_class_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig_class_distribution.png")

# --- 图2: 交易金额分布 ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.7, label='正常交易', color='#4CAF50', density=True)
ax.hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.7, label='欺诈交易', color='#F44336', density=True)
ax.set_xlabel('交易金额（欧元）', fontsize=12)
ax.set_ylabel('概率密度', fontsize=12)
ax.set_title('交易金额分布对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim([0, 500])
plt.tight_layout()
plt.savefig('fig_amount_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig_amount_distribution.png")

# --- 图3: 交易时间分布 ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(df[df['Class'] == 0]['Time'] / 3600, bins=48, alpha=0.7, label='正常交易', color='#4CAF50', density=True)
ax.hist(df[df['Class'] == 1]['Time'] / 3600, bins=48, alpha=0.7, label='欺诈交易', color='#F44336', density=True)
ax.set_xlabel('时间（小时）', fontsize=12)
ax.set_ylabel('概率密度', fontsize=12)
ax.set_title('交易时间分布对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('fig_time_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig_time_distribution.png")

# --- 图4: 金额vs时间散点图 ---
fig, ax = plt.subplots(figsize=(9, 6))
normal_sample = df[df['Class'] == 0].sample(n=len(fraud), random_state=42)
ax.scatter(fraud['Time'] / 3600, fraud['Amount'], c='#F44336', alpha=0.6, s=15, label='欺诈交易')
ax.scatter(normal_sample['Time'] / 3600, normal_sample['Amount'], c='#4CAF50', alpha=0.3, s=15, label='正常交易（采样）')
ax.set_xlabel('时间（小时）', fontsize=12)
ax.set_ylabel('交易金额（欧元）', fontsize=12)
ax.set_title('交易金额与时间散点图', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('fig_amount_time_scatter.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig_amount_time_scatter.png")

# --- 图5: PCA特征箱线图 ---
fig, ax = plt.subplots(figsize=(10, 6))
pca_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
plot_data = pd.melt(pd.concat([
    df[df['Class'] == 0][pca_features].assign(Class='正常').head(5000),
    df[df['Class'] == 1][pca_features].assign(Class='欺诈')
]), id_vars=['Class'], var_name='特征', value_name='值')
sns.boxplot(data=plot_data, x='特征', y='值', hue='Class',
            palette={'正常': '#4CAF50', '欺诈': '#F44336'}, ax=ax)
ax.set_title('PCA特征分布对比（V1-V8）', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('fig_pca_boxplot.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig_pca_boxplot.png")

# --- 图6: 特征相关性热力图 ---
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df.drop(columns=['Time']).corr()
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
            annot=False, linewidths=0.3, ax=ax, cbar_kws={'shrink': 0.7})
ax.set_title('特征相关性热力图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig_correlation_heatmap.png")

# --- 图7: 混淆矩阵 ---
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测正常', '预测欺诈'],
            yticklabels=['实际正常', '实际欺诈'],
            annot_kws={'size': 18}, ax=ax)
ax.set_title('混淆矩阵', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig_confusion_matrix.png")

# --- 图8: 模型对比 ---
models = {
    '决策树': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
    '逻辑回归': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    '随机森林': rf_best
}
model_names = []
model_f1 = []
model_auc = []
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred_m = model.predict(X_test)
    y_prob_m = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    f1_m = f1_score(y_test, y_pred_m)
    auc_m = roc_auc_score(y_test, y_prob_m) if y_prob_m is not None else 0
    model_names.append(name)
    model_f1.append(f1_m)
    model_auc.append(auc_m)

fig, ax = plt.subplots(figsize=(9, 6))
x_pos = np.arange(len(model_names))
width = 0.35
bars1 = ax.bar(x_pos - width/2, model_f1, width, label='F1-Score', color='#4CAF50')
bars2 = ax.bar(x_pos + width/2, model_auc, width, label='AUC-ROC', color='#2196F3')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, fontsize=12)
ax.set_ylim([0, 1.1])
ax.set_ylabel('分数', fontsize=12)
ax.set_title('模型性能对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('fig_model_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig_model_comparison.png")

# --- 图9: 特征重要性 ---
importances = rf_best.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[-15:][::-1]
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(range(15), importances[indices], color='steelblue')
ax.set_yticks(range(15))
ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
ax.set_xlabel('重要性分数', fontsize=12)
ax.set_title('特征重要性排序（Top-15）', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('fig_feature_importance.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig_feature_importance.png")

# --- 图10: 阈值分析 ---
thresholds = np.arange(0.1, 1.0, 0.05)
th_precisions = []
th_recalls = []
th_f1s = []
for th in thresholds:
    y_pred_th = (y_prob >= th).astype(int)
    th_precisions.append(precision_score(y_test, y_pred_th, zero_division=0))
    th_recalls.append(recall_score(y_test, y_pred_th, zero_division=0))
    th_f1s.append(f1_score(y_test, y_pred_th, zero_division=0))

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(thresholds, th_precisions, 'b-', linewidth=2, label='精确率')
ax.plot(thresholds, th_recalls, 'r-', linewidth=2, label='召回率')
ax.plot(thresholds, th_f1s, 'g--', linewidth=2, label='F1-Score')
best_th = thresholds[np.argmax(th_f1s)]
ax.axvline(x=best_th, color='gray', linestyle=':', linewidth=1, label=f'最优阈值={best_th:.2f}')
ax.set_xlabel('分类阈值', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('不同阈值下的性能变化', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig('fig_threshold_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig_threshold_analysis.png")

# =============================================================================
# 5. 生成2张组合图（附录用）
# =============================================================================
print("生成组合图...")

# --- EDA组合图 ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0, 0].pie([fraud_count[0], fraud_count[1]], labels=['正常', '欺诈'],
               autopct='%1.4f%%', colors=['#4CAF50', '#F44336'], explode=(0, 0.1))
axes[0, 0].set_title('(a) 交易类别分布', fontsize=12, fontweight='bold')

axes[0, 1].hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.7, label='正常', color='#4CAF50', density=True)
axes[0, 1].hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.7, label='欺诈', color='#F44336', density=True)
axes[0, 1].set_xlabel('交易金额'); axes[0, 1].set_ylabel('密度')
axes[0, 1].set_title('(b) 交易金额分布对比', fontsize=12, fontweight='bold')
axes[0, 1].legend(); axes[0, 1].set_xlim([0, 500])

axes[0, 2].hist(df[df['Class'] == 0]['Time'] / 3600, bins=48, alpha=0.7, label='正常', color='#4CAF50', density=True)
axes[0, 2].hist(df[df['Class'] == 1]['Time'] / 3600, bins=48, alpha=0.7, label='欺诈', color='#F44336', density=True)
axes[0, 2].set_xlabel('时间 (小时)'); axes[0, 2].set_ylabel('密度')
axes[0, 2].set_title('(c) 交易时间分布对比', fontsize=12, fontweight='bold')
axes[0, 2].legend()

normal_sample = df[df['Class'] == 0].sample(n=len(fraud), random_state=42)
axes[1, 0].scatter(fraud['Time'] / 3600, fraud['Amount'], c='#F44336', alpha=0.6, s=10, label='欺诈')
axes[1, 0].scatter(normal_sample['Time'] / 3600, normal_sample['Amount'], c='#4CAF50', alpha=0.3, s=10, label='正常(采样)')
axes[1, 0].set_xlabel('时间 (小时)'); axes[1, 0].set_ylabel('交易金额')
axes[1, 0].set_title('(d) 交易金额 vs 时间', fontsize=12, fontweight='bold')
axes[1, 0].legend()

sns.boxplot(data=plot_data, x='特征', y='值', hue='Class',
            palette={'正常': '#4CAF50', '欺诈': '#F44336'}, ax=axes[1, 1])
axes[1, 1].set_title('(e) PCA特征分布对比 (V1-V8)', fontsize=12, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)

sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
            annot=False, linewidths=0.3, ax=axes[1, 2], cbar_kws={'shrink': 0.7})
axes[1, 2].set_title('(f) 特征相关性热力图', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ eda_visualization.png")

# --- Results组合图 ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测正常', '预测欺诈'],
            yticklabels=['实际正常', '实际欺诈'],
            annot_kws={'size': 16}, ax=axes[0, 0])
axes[0, 0].set_title('(a) 混淆矩阵', fontsize=12, fontweight='bold')

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'Random Forest (AUC = {roc_auc:.4f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
axes[0, 1].set_xlabel('假正率 (False Positive Rate)')
axes[0, 1].set_ylabel('真正率 (True Positive Rate)')
axes[0, 1].set_title('(b) ROC曲线', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc='lower right')

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
axes[0, 2].plot(recall_curve, precision_curve, 'r-', linewidth=2, label=f'AP = {pr_auc:.4f}')
axes[0, 2].axhline(y=fraud_count[1]/len(df), color='k', linestyle='--', linewidth=1, label='随机水平')
axes[0, 2].set_xlabel('召回率 (Recall)')
axes[0, 2].set_ylabel('精确率 (Precision)')
axes[0, 2].set_title('(c) Precision-Recall曲线', fontsize=12, fontweight='bold')
axes[0, 2].legend(loc='upper right')

axes[1, 0].barh(range(15), importances[indices], color='steelblue')
axes[1, 0].set_yticks(range(15))
axes[1, 0].set_yticklabels([feature_names[i] for i in indices])
axes[1, 0].set_xlabel('重要性分数')
axes[1, 0].set_title('(d) 特征重要性 Top-15', fontsize=12, fontweight='bold')
axes[1, 0].invert_yaxis()

bars1 = axes[1, 1].bar(x_pos - width/2, model_f1, width, label='F1-Score', color='#4CAF50')
bars2 = axes[1, 1].bar(x_pos + width/2, model_auc, width, label='AUC-ROC', color='#2196F3')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(model_names)
axes[1, 1].set_ylim([0, 1])
axes[1, 1].set_ylabel('分数')
axes[1, 1].set_title('(e) 模型性能对比', fontsize=12, fontweight='bold')
axes[1, 1].legend()
for bar in bars1:
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

axes[1, 2].plot(thresholds, th_precisions, 'b-', linewidth=2, label='精确率')
axes[1, 2].plot(thresholds, th_recalls, 'r-', linewidth=2, label='召回率')
axes[1, 2].plot(thresholds, th_f1s, 'g--', linewidth=2, label='F1-Score')
axes[1, 2].axvline(x=best_th, color='gray', linestyle=':', linewidth=1, label=f'最优阈值={best_th:.2f}')
axes[1, 2].set_xlabel('分类阈值')
axes[1, 2].set_ylabel('分数')
axes[1, 2].set_title('(f) 不同阈值的性能变化', fontsize=12, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('results_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ results_visualization.png")

print("\n所有图表生成完成！")
print(f"输出目录: {output_dir}")
