from Utils.metrics import jaccard_score_background, jaccard_score_true_class, jaccard_score, dice_coef
import tensorflow as tf

def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_score(y_true, y_pred)

def weighted_jaccard_loss(y_true, y_pred):
    return 3 * (1 - jaccard_score_true_class(y_true, y_pred)) + (1 - jaccard_score_background(y_true, y_pred))

def combined_loss(y_true, y_pred):
    return 5 * ssim_loss(y_true, y_pred) + jaccard_loss(y_true,y_pred)