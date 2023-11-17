import torch
from sklearn.metrics import f1_score

# f1 score 
def cal_accuracy(logits, labels, threshold=0.9):
    # Convert logits to binary predictions by thresholding at the specified threshold
    predicted_labels = (logits > threshold).int()
    f1score = f1_score(labels, predicted_labels, average='micro')
    return f1score

# def cal_accuracy(logits, labels):
    # # Convert logits to binary predictions by thresholding at 0.5
    # predicted_labels = (logits > 0.9).int()
    # correct_predictions = torch.eq(predicted_labels, labels)
    # accuracy = correct_predictions.float().mean()
    # return accuracy.item()
