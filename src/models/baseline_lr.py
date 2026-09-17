"""
Baseline: logistic regression on the CURRENT state only (no history). The
point is to isolate one variable — does remembering the trajectory help —
by comparing against a model with equivalent per-window features but zero
memory (techsoln.md sec 8).
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class BaselineLogReg:
    def __init__(self, max_iter: int = 1000):
        self.inf_model = LogisticRegression(max_iter=max_iter)
        self.stage_model = LogisticRegression(max_iter=max_iter, multi_class="auto")

    def fit(self, X_last_state, y_inf, y_stage):
        self.inf_model.fit(X_last_state, y_inf)
        self.stage_model.fit(X_last_state, y_stage)
        return self

    def predict_infiltration(self, X_last_state):
        return self.inf_model.predict(X_last_state)

    def predict_stage(self, X_last_state):
        return self.stage_model.predict(X_last_state)


def infiltration_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }
