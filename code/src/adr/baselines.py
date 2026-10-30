"""Classical ML baselines on fingerprints / descriptors."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline(name: str, seed=0, scale_pos_weight=1.0, n_jobs=-1):
    name = name.lower()
    if name == "logreg":
        return make_pipeline(StandardScaler(with_mean=False),
                             LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced", random_state=seed))
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=2,
                                      n_jobs=n_jobs, random_state=seed)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True,
                                                   class_weight="balanced", random_state=seed))
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.6, scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                             n_jobs=n_jobs, random_state=seed, verbosity=0)
    raise ValueError(name)


def fit_predict(name, X_tr, y_tr, X_te, seed=0):
    pos = max(int(y_tr.sum()), 1); neg = len(y_tr) - pos
    clf = make_baseline(name, seed=seed, scale_pos_weight=neg / pos)
    clf.fit(X_tr, y_tr)
    return clf, clf.predict_proba(X_te)[:, 1]
