import torch
from sklearn.metrics import f1_score

def cal_accuracy(logits, labels, threshold=0.5):
    # Convert logits to binary predictions by thresholding at the specified threshold
    predicted_labels = (logits > threshold).int()

    # Calculate the F1 score for each label and then take the mean
    f1_scores = []
    for i in range(labels.shape[1]):
        f1 = f1_score(labels[:, i], predicted_labels[:, i])
        f1_scores.append(f1)

    # Calculate the average F1 score across all labels
    mean_f1_score = sum(f1_scores) / len(f1_scores)

    return mean_f1_score


# def cal_accuracy(logits, labels):
#     # Convert logits to binary predictions by thresholding at 0.5
#     predicted_labels = (logits > 0.5).int()

#     # Calculate element-wise equality between predicted and target labels
#     correct_predictions = torch.eq(predicted_labels, labels)

#     # Calculate the accuracy by taking the mean of correct predictions
#     accuracy = correct_predictions.float().mean()

#     return accuracy.item()

#     # predicts = torch.argmax(logits, dim=1)
#     # acc = torch.mean((predicts == labels).float())
#     # return acc