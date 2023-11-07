import torch


def cal_accuracy(logits, labels):
    # Convert logits to binary predictions by thresholding at 0.5
    predicted_labels = (logits > 0.5).int()

    # Calculate element-wise equality between predicted and target labels
    correct_predictions = torch.eq(predicted_labels, labels)

    # Calculate the accuracy by taking the mean of correct predictions
    accuracy = correct_predictions.float().mean()

    return accuracy.item()

    # predicts = torch.argmax(logits, dim=1)
    # acc = torch.mean((predicts == labels).float())
    # return acc