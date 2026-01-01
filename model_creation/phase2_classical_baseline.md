# Phase 2: Classical ML Baseline (Weeks 3-4)
## Quantum-Enhanced Cryptanalysis System

**Duration:** 2 weeks  
**Goal:** Train ensemble of classical models and establish performance baseline

---

## OBJECTIVES

1. Implement 7 base learners (XGBoost, RF, LightGBM, SVM, NN, CatBoost, LogReg)
2. Hyperparameter optimization (Bayesian search)
3. Create stacking ensemble
4. Evaluate all 5 tasks
5. Document baseline performance

---

## INPUT SPECIFICATIONS

**Source:** `d:\Dream\AIRAWAT\model_creation\data\processed\`

```
X_train.csv - 30,000 × 65 features
y_train.csv - 5 task targets
X_val.csv - 10,000 × 65
y_val.csv
```

---

## OUTPUT SPECIFICATIONS

**Target:** `d:\Dream\AIRAWAT\model_creation\models\classical_baseline\`

### Trained Models (per task)
```
task1_xgboost.pkl
task1_random_forest.pkl
task1_lightgbm.pkl
task1_catboost.pkl
task1_svm.pkl
task1_neural_network.h5
task1_logistic_regression.pkl
task1_stacked_ensemble.pkl
```

### Performance Reports
```
reports/classical_baseline_report.pdf:
- Per-model performance comparison
- Confusion matrices
- ROC/PR curves
- Feature importance analysis
- Hyperparameter search results
```

---

## WEEK 3: BASE LEARNER IMPLEMENTATION

### Day 1-2: Task 1 - Attack Classification

**Implementation:**
```python
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import optuna

# XGBoost
def train_xgboost_task1(X_train, y_train, X_val, y_val):
    model = XGBClassifier(
        max_depth=7,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=10
    )
    
    return model

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)

# LightGBM  
lgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    num_leaves=63,
    max_depth=7,
    random_state=42
)
lgbm_model.fit(X_train, y_train)
```

**Evaluation:**
```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Predictions
xgb_pred = xgb_model.predict(X_val)
rf_pred = rf_model.predict(X_val)
lgbm_pred = lgbm_model.predict(X_val)

# Metrics
print(f"XGBoost Accuracy: {accuracy_score(y_val, xgb_pred):.4f}")
print(f"XGBoost F1 (macro): {f1_score(y_val, xgb_pred, average='macro'):.4f}")

# Target: ≥86% accuracy baseline
```

**Deliverable:** `src/classical/base_learners.py`

---

### Day 3-4: Task 2 - Attack Success Prediction

**Handle Class Imbalance:**
```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# Check imbalance
class_counts = y_train['attack_success'].value_counts()
ratio = class_counts.min() / class_counts.max()
print(f"Imbalance ratio: {ratio:.3f}")

if ratio < 0.3:
    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train['attack_success'])
else:
    # Use class weights
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train['attack_success'])
    
    xgb_task2 = XGBClassifier(
        scale_pos_weight=class_weights[1] / class_weights[0],
        max_depth=7,
        learning_rate=0.1
    )
```

**SVM with RBF Kernel:**
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,  # Enable predict_proba
    random_state=42
)
svm_model.fit(X_train_scaled, y_train['attack_success'])
```

**Evaluation:**
```python
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

# ROC-AUC (primary metric)
y_pred_proba = xgb_task2.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val['attack_success'], y_pred_proba)
print(f"ROC-AUC: {auc:.4f}")  # Target: ≥0.80

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_val['attack_success'], y_pred_proba)
```

**Deliverable:** `src/classical/task2_success_prediction.py`

---

### Day 5: Task 3-5 (Simplified Baselines)

**Task 3: Algorithm Identification**
```python
# Use same XGBoost + RF setup
xgb_task3 = XGBClassifier(max_depth=7, n_estimators=300)
xgb_task3.fit(X_train, y_train['algorithm_name'])

# Metrics
from sklearn.metrics import top_k_accuracy_score

top1_acc = accuracy_score(y_val['algorithm_name'], xgb_task3.predict(X_val))
top2_acc = top_k_accuracy_score(y_val['algorithm_name'], xgb_task3.predict_proba(X_val), k=2)

print(f"Top-1 Accuracy: {top1_acc:.4f}")  # Target: ≥85%
print(f"Top-2 Accuracy: {top2_acc:.4f}")  # Target: ≥95%
```

---

## WEEK 4: HYPERPARAMETER TUNING & ENSEMBLE

### Day 6-7: Bayesian Optimization

**Optuna Setup:**
```python
import optuna
from sklearn.model_selection import cross_val_score

def objective_xgboost_task1(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 500, step=50),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }
    
    model = XGBClassifier(**params, random_state=42, eval_metric='mlogloss')
    
    # 5-fold CV
    score = cross_val_score(model, X_train, y_train['attack_category'], cv=5, scoring='f1_macro').mean()
    
    return score

# Run optimization
study = optuna.create_study(direction='maximize', study_name='xgboost_task1')
study.optimize(objective_xgboost_task1, n_trials=100, n_jobs=4)

print(f"Best F1 score: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Save best hyperparameters
import json
with open('models/best_params_task1_xgboost.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
```

**Deliverable:** `src/classical/hyperopt.py`

---

### Day 8-9: Stacking Ensemble

**Level 0: Base Learner Predictions**
```python
# Get predictions from all 7 learners
models = {
    'xgboost': xgb_model,
    'rf': rf_model,
    'lightgbm': lgbm_model,
    'catboost': catboost_model,
    'svm': svm_model,
    'nn': nn_model,
    'logreg': logreg_model
}

# Collect probability predictions (for multi-class)
n_classes = len(np.unique(y_train['attack_category']))
level0_features_train = []
level0_features_val = []

for name, model in models.items():
    # Get probabilities: shape (n_samples, n_classes)
    train_proba = model.predict_proba(X_train)
    val_proba = model.predict_proba(X_val)
    
    level0_features_train.append(train_proba)
    level0_features_val.append(val_proba)

# Stack: (n_samples, 7 models × n_classes features)
X_train_level1 = np.hstack(level0_features_train)  # Shape: (30000, 7×5=35)
X_val_level1 = np.hstack(level0_features_val)
```

**Level 1: Meta-Learner**
```python
# Train meta-learner on Level 0 predictions
meta_learner = XGBClassifier(
    max_depth=3,  # Shallow to avoid overfitting
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

meta_learner.fit(X_train_level1, y_train['attack_category'])

# Final predictions
final_pred = meta_learner.predict(X_val_level1)
final_accuracy = accuracy_score(y_val['attack_category'], final_pred)

print(f"Stacked Ensemble Accuracy: {final_accuracy:.4f}")
# Target: Improvement over best base learner
```

**Save Ensemble:**
```python
import pickle

ensemble_package = {
    'base_learners': models,
    'meta_learner': meta_learner,
    'feature_cols': feature_cols,
    'scaler': scaler  # if used for SVM/NN
}

with open('models/classical_baseline/task1_stacked_ensemble.pkl', 'wb') as f:
    pickle.dump(ensemble_package, f)
```

**Deliverable:** `src/classical/ensemble.py`

---

### Day 10: Performance Report

**Generate Comprehensive Report:**
```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Performance comparison table
results = {
    'Model': ['XGBoost', 'RF', 'LightGBM', 'CatBoost', 'SVM', 'NN', 'LogReg', 'Stacked'],
    'Accuracy': [...],  # Fill from evaluations
    'Macro F1': [...],
    'Weighted F1': [...]
}

results_df = pd.DataFrame(results)
print(results_df.to_latex())  # For PDF report

# Confusion matrix
cm = confusion_matrix(y_val['attack_category'], final_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Task 1: Attack Classification - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('results/plots/task1_confusion_matrix.png', dpi=300)

# Feature importance (SHAP)
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
plt.savefig('results/plots/task1_feature_importance.png', dpi=300)
```

**Report Structure:**
```
reports/classical_baseline_report.pdf:
1. Executive Summary (1 page)
   - Best model per task
   - Performance vs targets
   
2. Task 1: Attack Classification (3 pages)
   - Model comparison table
   - Confusion matrix
   - Per-class precision/recall
   - Feature importance
   
3. Task 2: Attack Success Prediction (3 pages)
   - ROC-AUC curves
   - Precision-Recall curves
   - Class imbalance handling results
   
4. Tasks 3-5: Summary (2 pages)
   
5. Ensemble Analysis (2 pages)
   - Stacking performance gains
   - Meta-learner analysis
   
6. Hyperparameter Search (2 pages)
   - Optuna optimization curves
   - Best parameters per model
   
7. Conclusions & Next Steps (1 page)
```

**Deliverable:** `reports/classical_baseline_report.pdf`

---

## WEEK-END DELIVERABLES

### Code
- `src/classical/base_learners.py` - All 7 learner implementations
- `src/classical/ensemble.py` - Stacking logic
- `src/classical/hyperopt.py` - Bayesian optimization
- `src/classical/evaluator.py` - Metrics & plotting

### Models (40 files total: 8 models × 5 tasks)
- `models/classical_baseline/task{1-5}_{model_name}.pkl`
- `models/best_params_*.json`

### Reports
- `reports/classical_baseline_report.pdf`
- `reports/hyperparameter_search_results.csv`

---

## SUCCESS CRITERIA

✓ Task 1: ≥86% accuracy (baseline target)  
✓ Task 2: ≥0.80 ROC-AUC  
✓ Task 3: ≥85% accuracy  
✓ All models saved with hyperparameters  
✓ Stacking ensemble improves over best single model  
✓ Feature importance documented  
✓ Cross-validation performed (5-fold)  
✓ Report generated with all visualizations

---

**Next Phase:** [Phase 3: Quantum Circuit Design](./phase3_quantum_circuits.md)
