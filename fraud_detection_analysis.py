# -*- coding: utf-8 -*-
"""
=============================================================================
基于随机森林的信用卡欺诈检测 - 完整实验代码
=============================================================================
数据集: Kaggle Credit Card Fraud Detection (creditcard.csv)
目标: 使用随机森林算法实现高精度的信用卡欺诈交易检测

作者: [请填写]
日期: 2026年6月
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 数据加载与探索性分析 (EDA)
# =============================================================================
print("=" * 60)
print("1. 数据加载与探索性分析")
print("=" * 60)

# 加载数据
df = pd.read_csv('creditcard.csv')
print(f"数据集维度: {df.shape[0]} 条交易, {df.shape[1]} 个特征")
print(f"\n数据预览:\n{df.head()}")
print(f"\n数据类型:\n{df.dtypes.value_counts()}")
print(f"\n缺失值统计:\n{df.isnull().sum().sum()} 个缺失值")

# 基本统计信息
print("\n--- 基本统计描述 (Amount) ---")
print(df['Amount'].describe())

print("\n--- 基本统计描述 (Time) ---")
print(df['Time'].describe())

# 类别分布
fraud_count = df['Class'].value_counts()
print(f"\n类别分布:\n正常交易: {fraud_count[0]} ({fraud_count[0]/len(df)*100:.2f}%)")
print(f"欺诈交易: {fraud_count[1]} ({fraud_count[1]/len(df)*100:.4f}%)")
print(f"不平衡比例: {fraud_count[0]/fraud_count[1]:.1f}:1")

# =============================================================================
# 2. 数据可视化
# =============================================================================
print("\n" + "=" * 60)
print("2. 数据可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) 类别分布饼图
axes[0, 0].pie([fraud_count[0], fraud_count[1]], labels=['正常', '欺诈'],
               autopct='%1.4f%%', colors=['#4CAF50', '#F44336'], explode=(0, 0.1))
axes[0, 0].set_title('(a) 交易类别分布', fontsize=12, fontweight='bold')

# (b) 交易金额分布
axes[0, 1].hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.7, label='正常', color='#4CAF50', density=True)
axes[0, 1].hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.7, label='欺诈', color='#F44336', density=True)
axes[0, 1].set_xlabel('交易金额')
axes[0, 1].set_ylabel('密度')
axes[0, 1].set_title('(b) 交易金额分布对比', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].set_xlim([0, 500])

# (c) 交易时间分布
axes[0, 2].hist(df[df['Class'] == 0]['Time'] / 3600, bins=48, alpha=0.7, label='正常', color='#4CAF50', density=True)
axes[0, 2].hist(df[df['Class'] == 1]['Time'] / 3600, bins=48, alpha=0.7, label='欺诈', color='#F44336', density=True)
axes[0, 2].set_xlabel('时间 (小时)')
axes[0, 2].set_ylabel('密度')
axes[0, 2].set_title('(c) 交易时间分布对比', fontsize=12, fontweight='bold')
axes[0, 2].legend()

# (d) 欺诈交易金额 vs 时间散点图
fraud = df[df['Class'] == 1]
normal_sample = df[df['Class'] == 0].sample(n=len(fraud), random_state=42)
axes[1, 0].scatter(fraud['Time'] / 3600, fraud['Amount'], c='#F44336', alpha=0.6, s=10, label='欺诈')
axes[1, 0].scatter(normal_sample['Time'] / 3600, normal_sample['Amount'], c='#4CAF50', alpha=0.3, s=10, label='正常(采样)')
axes[1, 0].set_xlabel('时间 (小时)')
axes[1, 0].set_ylabel('交易金额')
axes[1, 0].set_title('(d) 交易金额 vs 时间', fontsize=12, fontweight='bold')
axes[1, 0].legend()

# (e) 部分PCA特征的箱线图
pca_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
plot_data = pd.melt(pd.concat([
    df[df['Class'] == 0][pca_features].assign(Class='正常').head(5000),
    df[df['Class'] == 1][pca_features].assign(Class='欺诈')
]), id_vars=['Class'], var_name='特征', value_name='值')
sns.boxplot(data=plot_data, x='特征', y='值', hue='Class',
            palette={'正常': '#4CAF50', '欺诈': '#F44336'}, ax=axes[1, 1])
axes[1, 1].set_title('(e) PCA特征分布对比 (V1-V8)', fontsize=12, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)

# (f) 特征相关性热力图
corr_matrix = df.drop(columns=['Time']).corr()
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
            annot=False, linewidths=0.3, ax=axes[1, 2],
            cbar_kws={'shrink': 0.7})
axes[1, 2].set_title('(f) 特征相关性热力图', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: eda_visualization.png")

# =============================================================================
# 3. 数据预处理
# =============================================================================
print("\n" + "=" * 60)
print("3. 数据预处理")
print("=" * 60)

# 分离特征和标签
X = df.drop(columns=['Class'])
y = df['Class']

# 时间特征转换：将秒转换为小时
X['Time'] = X['Time'] / 3600

# 特征标准化 (RobustScaler 对异常值更鲁棒)
scaler_amount = RobustScaler()
scaler_time = RobustScaler()
scaler_others = StandardScaler()

X['Amount'] = scaler_amount.fit_transform(X[['Amount']])
X['Time'] = scaler_time.fit_transform(X[['Time']])
other_cols = [c for c in X.columns if c not in ['Amount', 'Time']]
X[other_cols] = scaler_others.fit_transform(X[other_cols])

print(f"标准化后特征维度: {X.shape}")
print(f"标准化后Amount均值: {X['Amount'].mean():.4f}, 标准差: {X['Amount'].std():.4f}")

# 数据集划分 (分层抽样确保训练/测试集中欺诈比例一致)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
print(f"训练集欺诈比例: {y_train.sum()/len(y_train)*100:.4f}%")
print(f"测试集欺诈比例: {y_test.sum()/len(y_test)*100:.4f}%")

# =============================================================================
# 4. 处理类别不平衡 (SMOTE过采样)
# =============================================================================
print("\n" + "=" * 60)
print("4. 处理类别不平衡 (SMOTE)")
print("=" * 60)

smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"SMOTE前训练集大小: {len(y_train)}, 欺诈样本: {y_train.sum()}")
print(f"SMOTE后训练集大小: {len(y_train_smote)}, 欺诈样本: {y_train_smote.sum()}")
print(f"SMOTE后欺诈比例: {y_train_smote.sum()/len(y_train_smote)*100:.2f}%")

# =============================================================================
# 5. 随机森林模型训练与超参数调优
# =============================================================================
print("\n" + "=" * 60)
print("5. 随机森林模型训练与调优")
print("=" * 60)

# 基础模型
rf_base = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_base.fit(X_train_smote, y_train_smote)
y_pred_base = rf_base.predict(X_test)
print(f"基础模型测试集F1-score: {f1_score(y_test, y_pred_base):.4f}")
print(f"基础模型测试集AUC-ROC: {roc_auc_score(y_test, rf_base.predict_proba(X_test)[:, 1]):.4f}")

# 网格搜索调优
print("\n--- 网格搜索超参数调优 ---")
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
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train_smote, y_train_smote)

print(f"\n最佳参数: {rf_grid.best_params_}")
print(f"最佳交叉验证F1-score: {rf_grid.best_score_:.4f}")

# 最优模型
rf_best = rf_grid.best_estimator_

# =============================================================================
# 6. 模型评估
# =============================================================================
print("\n" + "=" * 60)
print("6. 模型评估")
print("=" * 60)

y_pred = rf_best.predict(X_test)
y_prob = rf_best.predict_proba(X_test)[:, 1]

print("\n--- 分类报告 ---")
print(classification_report(y_test, y_pred, target_names=['正常交易', '欺诈交易']))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"\n--- 混淆矩阵 ---")
print(f"TP={cm[1,1]}, FP={cm[0,1]}, TN={cm[0,0]}, FN={cm[1,0]}")

# 综合指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print(f"\n--- 综合指标汇总 ---")
print(f"准确率 (Accuracy):   {accuracy:.4f}")
print(f"精确率 (Precision):  {precision:.4f}")
print(f"召回率 (Recall):     {recall:.4f}")
print(f"F1分数:              {f1:.4f}")
print(f"AUC-ROC:             {roc_auc:.4f}")
print(f"AUC-PR (AP):         {pr_auc:.4f}")

# =============================================================================
# 7. 混淆矩阵与ROC曲线可视化
# =============================================================================
print("\n" + "=" * 60)
print("7. 结果可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# (a) 混淆矩阵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测正常', '预测欺诈'],
            yticklabels=['实际正常', '实际欺诈'],
            annot_kws={'size': 16}, ax=axes[0, 0])
axes[0, 0].set_title('(a) 混淆矩阵', fontsize=12, fontweight='bold')

# (b) ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'Random Forest (AUC = {roc_auc:.4f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
axes[0, 1].set_xlabel('假正率 (False Positive Rate)')
axes[0, 1].set_ylabel('真正率 (True Positive Rate)')
axes[0, 1].set_title('(b) ROC曲线', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc='lower right')

# (c) Precision-Recall曲线
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
axes[0, 2].plot(recall_curve, precision_curve, 'r-', linewidth=2, label=f'AP = {pr_auc:.4f}')
axes[0, 2].axhline(y=fraud_count[1]/len(df), color='k', linestyle='--', linewidth=1, label='随机水平')
axes[0, 2].set_xlabel('召回率 (Recall)')
axes[0, 2].set_ylabel('精确率 (Precision)')
axes[0, 2].set_title('(c) Precision-Recall曲线', fontsize=12, fontweight='bold')
axes[0, 2].legend(loc='upper right')

# (d) 特征重要性 Top-15
importances = rf_best.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[-15:][::-1]
axes[1, 0].barh(range(15), importances[indices], color='steelblue')
axes[1, 0].set_yticks(range(15))
axes[1, 0].set_yticklabels([feature_names[i] for i in indices])
axes[1, 0].set_xlabel('重要性分数')
axes[1, 0].set_title('(d) 特征重要性 Top-15', fontsize=12, fontweight='bold')
axes[1, 0].invert_yaxis()

# (e) 模型对比
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
    print(f"{name}: F1={f1_m:.4f}, AUC={auc_m:.4f}")

x_pos = np.arange(len(model_names))
width = 0.35
bars1 = axes[1, 1].bar(x_pos - width/2, model_f1, width, label='F1-Score', color='#4CAF50')
bars2 = axes[1, 1].bar(x_pos + width/2, model_auc, width, label='AUC-ROC', color='#2196F3')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(model_names)
axes[1, 1].set_ylim([0, 1])
axes[1, 1].set_ylabel('分数')
axes[1, 1].set_title('(e) 模型性能对比', fontsize=12, fontweight='bold')
axes[1, 1].legend()

# 在柱状图上标注数值
for bar in bars1:
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

# (f) 不同阈值下的精确率-召回率权衡
thresholds = np.arange(0.1, 1.0, 0.05)
th_precisions = []
th_recalls = []
th_f1s = []
for th in thresholds:
    y_pred_th = (y_prob >= th).astype(int)
    th_precisions.append(precision_score(y_test, y_pred_th, zero_division=0))
    th_recalls.append(recall_score(y_test, y_pred_th, zero_division=0))
    th_f1s.append(f1_score(y_test, y_pred_th, zero_division=0))

axes[1, 2].plot(thresholds, th_precisions, 'b-', linewidth=2, label='精确率')
axes[1, 2].plot(thresholds, th_recalls, 'r-', linewidth=2, label='召回率')
axes[1, 2].plot(thresholds, th_f1s, 'g--', linewidth=2, label='F1-Score')
best_th = thresholds[np.argmax(th_f1s)]
axes[1, 2].axvline(x=best_th, color='gray', linestyle=':', linewidth=1, label=f'最优阈值={best_th:.2f}')
axes[1, 2].set_xlabel('分类阈值')
axes[1, 2].set_ylabel('分数')
axes[1, 2].set_title('(f) 不同阈值的性能变化', fontsize=12, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('results_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: results_visualization.png")

# =============================================================================
# 8. 特征重要性详细分析
# =============================================================================
print("\n" + "=" * 60)
print("8. 特征重要性详细分析")
print("=" * 60)

feature_importance_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': importances
}).sort_values('重要性', ascending=False)

print("\nTop-20 最重要特征:")
print(feature_importance_df.head(20).to_string(index=False))

# =============================================================================
# 9. 交叉验证
# =============================================================================
print("\n" + "=" * 60)
print("9. 交叉验证评估")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'roc_auc': []
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_smote, y_train_smote)):
    X_tr, X_val = X_train_smote.iloc[train_idx], X_train_smote.iloc[val_idx]
    y_tr, y_val = y_train_smote.iloc[train_idx], y_train_smote.iloc[val_idx]

    rf_cv = RandomForestClassifier(**rf_grid.best_params_, random_state=42, n_jobs=-1)
    rf_cv.fit(X_tr, y_tr)
    y_pred_cv = rf_cv.predict(X_val)
    y_prob_cv = rf_cv.predict_proba(X_val)[:, 1]

    cv_scores['accuracy'].append(accuracy_score(y_val, y_pred_cv))
    cv_scores['precision'].append(precision_score(y_val, y_pred_cv))
    cv_scores['recall'].append(recall_score(y_val, y_pred_cv))
    cv_scores['f1'].append(f1_score(y_val, y_pred_cv))
    cv_scores['roc_auc'].append(roc_auc_score(y_val, y_prob_cv))

print(f"\n5折交叉验证结果:")
for metric, scores in cv_scores.items():
    print(f"  {metric:12s}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# =============================================================================
# 10. 欺诈样例分析
# =============================================================================
print("\n" + "=" * 60)
print("10. 误分类案例分析")
print("=" * 60)

# 找出误分类样本
X_test_array = X_test.values
misclassified = X_test[y_test != y_pred]
mis_true = y_test[y_test != y_pred]
mis_pred = y_pred[y_test != y_pred]

fp_count = ((y_test == 0) & (y_pred == 1)).sum()  # 假阳性
fn_count = ((y_test == 1) & (y_pred == 0)).sum()  # 假阴性

print(f"假阳性 (误报): {fp_count} 个正常交易被误判为欺诈")
print(f"假阴性 (漏报): {fn_count} 个欺诈交易被漏过")

# 分析误分类样本的特征
if fn_count > 0:
    fn_samples = X_test[(y_test == 1) & (y_pred == 0)]
    print(f"\n漏报欺诈交易的平均金额: {fn_samples['Amount'].mean():.4f} (标准化后)")
    print(f"漏报欺诈交易的平均发生时间: {fn_samples['Time'].mean():.4f} (标准化后)")

print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
print("\n输出文件:")
print("  - eda_visualization.png     : 探索性数据分析可视化")
print("  - results_visualization.png : 模型评估结果可视化")
