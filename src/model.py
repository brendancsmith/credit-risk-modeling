#TODO: model.py

from xgboost import XGBClassifier

def create_model(**kwargs):

    defaults = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
    }

    params = defaults | kwargs

    # Initialize model
    xgb_clf = XGBClassifier(**params)

    return xgb_clf
