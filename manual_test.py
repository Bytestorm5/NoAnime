import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import cv2

model = keras.models.load_model('final_model')
preds = []
positives = []
negatives = []
for filename in os.listdir("output\\0"):
    f = os.path.join("output\\0", filename)
    img = cv2.imread(f)
    imgr = cv2.resize(img, (256, 256))
    imgr=np.expand_dims(imgr,axis=0)
    pred = model.predict(imgr)
    print(f"Predicted {pred} for {f}")
    preds.append([pred[0][0], 0])
    negatives.append(pred[0][0])

for filename in os.listdir("output\\1"):
    f = os.path.join("output\\1", filename)
    img = cv2.imread(f)
    imgr = cv2.resize(img, (256, 256))
    imgr=np.expand_dims(imgr,axis=0)
    pred = model.predict(imgr)
    print(f"Predicted {pred} for {f}")
    preds.append([pred[0][0], 1])
    positives.append(pred[0][0])

ths = np.linspace(0, 1, num=1000)
max_f1 = 0
max_th = 0
max_matrix = []
recalls = []
precisions = []
for th in ths:
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    for pred in preds:
        if pred[0] > th:
            if pred[1]:
                TP += 1.0
            else:
                FP += 1.0
        else:
            if pred[1]:
                FN += 1.0
            else:
                TN += 1.0
    recall = TP / (FN + TP) if FN + TP > 0 else 1
    precision = TP / (TP + FP) if TP + FP > 0 else 1

    recalls.append(recall)
    precisions.append(precision)

    rr = 1/(max(0.00001, recall))
    pr = 1/(max(0.00001, precision))    
    f1 = 1.0 / (np.average([rr, pr]))
    if f1 > max_f1:
        max_f1 = f1
        max_th = th
        max_matrix = [TP, TN, FP, FN]
print(f"\nBEST THRESHOLD: {max_th}")
print(f"""All predictions above {max_th} should be considered to be diagnostic, and all predictions below {max_th} should be considered as non-diagnostic.\n""")
print(f"BEST MATRIX: {str(max_matrix)}")
if max_matrix[0] + max_matrix[3] > 0:
    print(f"Recall: {max_matrix[0] / (max_matrix[0] + max_matrix[3])}")
if max_matrix[0] + max_matrix[2] > 0:
    print(f"Precision: {max_matrix[0] / (max_matrix[0] + max_matrix[2])}")
print(f"Accuracy: {(max_matrix[0] + max_matrix[1]) / sum(max_matrix)}")
print("-----------------------------")
print("Positive:")
print(f"Max: {max(positives)}\nMin:{min(positives)}\nAverage: {np.average(positives)}\nVariance: {np.var(positives)}")
print("------------------------------")
print("Negatives:")
print(f"Max: {max(negatives)}\nMin:{min(negatives)}\nAverage: {np.average(negatives)}\nVariance: {np.var(negatives)}")

plt.subplot(1, 2, 1)
plt.hist(positives)
plt.title("Positive Cases")
plt.subplot(1, 2, 2)
plt.hist(negatives)
plt.title("Negative Cases")
plt.show()

plt.plot(recalls, precisions)
plt.title('model precision vs. recall')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend("train", "validation")
plt.show()