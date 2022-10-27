from Utils.metrics import jaccard_score

def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_score(y_true, y_pred)