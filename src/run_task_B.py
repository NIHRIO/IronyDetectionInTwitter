from DataProcessor import DataProcessor
from MLP import MLP
import numpy as np

n_fold = 10
train_file = "data/SemEval2018-T3-taskB.txt"
test_file = "data/SemEval2018-T3_input_test_taskA.txt" # the test data is the same for both subtasks A and B
train_data, test_data = DataProcessor().process_data(train_file, test_file, load_saved_data=False)
k_fold_train, k_fold_valid = DataProcessor.split_kfolds(train_data, n_fold)

mlp_predict = None
mlp_f1_scores = []

for i in range(len(k_fold_train)):
    print("====================Fold %d=================" % (i + 1))
    mlp_pred_train, mlp_pred_valid, mlp_pred_test, mlp_f1_score = MLP().predict(k_fold_train[i], k_fold_valid[i], test_data, task_name="B")
    mlp_f1_scores.append(mlp_f1_score)
    if mlp_predict is None:
        mlp_predict = mlp_pred_test
    else:
        mlp_predict = np.column_stack((mlp_predict, mlp_pred_test))


file_out = open("predictions-taskB.txt", "w")
for i in range(len(mlp_predict)):
    if i > 0:
        pred_labels = mlp_predict[i]
        tmp = np.zeros(4)
        for lb in pred_labels:
            tmp[lb] += 1
        label = np.argmax(tmp)
        file_out.write("%d\n" % label)

file_out.close()


mlp_f1_scores = np.array(mlp_f1_scores)
print("Final mlp F1: %0.4f (+/- %0.4f)" % (mlp_f1_scores.mean(), mlp_f1_scores.std() * 2))
