import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
#import lightgbm as lgb
X = np.load('X_features_pca_all.npy')  # or X_features_pca.npy for just one payload
y = np.load('y_labels_all.npy')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate_model(name, model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print(f"\n{name} Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")

    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
evaluate_model("Logistic Regression", lr, X_val, y_val)

'''svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
evaluate_model("SVM (RBF Kernel)", svm, X_val, y_val)'''

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
evaluate_model("Random Forest", rf, X_val, y_val)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
evaluate_model("XGBoost", xgb_model, X_val, y_val)

from joblib import dump
#dump(xgb_model, 'final_xgb_model.joblib')  
dump(rf, 'final_model.joblib') 

# Compute ROC curves and AUC
from sklearn.metrics import roc_curve, auc

fpr_lr, tpr_lr, _ = roc_curve(y_val, y_prob_lr)
fpr_svm, tpr_svm, _ = roc_curve(y_val, y_prob_svm)
fpr_rf, tpr_rf, _ = roc_curve(y_val, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_prob_xgb)

auc_lr = auc(fpr_lr, tpr_lr)
auc_svm = auc(fpr_svm, tpr_svm)
auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.4f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (RBF) (AUC = {auc_svm:.4f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.4f})')

# Diagonal line
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison – Steganalysis Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
