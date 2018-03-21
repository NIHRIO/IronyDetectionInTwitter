from DataProcessor import DataProcessor
from MLP import MLP
import numpy as np

n_fold = 10
train_file = "data/SemEval2018-T3-taskA.txt"
test_file = "data/SemEval2018-T3_input_test_taskA.txt"

train_data, test_data = DataProcessor().process_data(train_file, test_file, load_saved_data=False)
k_fold_train, k_fold_valid = DataProcessor.split_kfolds(train_data, n_fold)

mlp_predict = None

mlp_f1_scores = []

for i in range(len(k_fold_train)):
    print("====================Fold %d=================" % (i + 1))
    _, _, mlp_pred_test, mlp_f1_score = MLP().predict(k_fold_train[i], k_fold_valid[i], test_data)
    mlp_f1_scores.append(mlp_f1_score)
    if mlp_predict is None:
        mlp_predict = mlp_pred_test
    else:
        mlp_predict = np.column_stack((mlp_predict, mlp_pred_test))

mlp_predict = np.average(mlp_predict, axis=1)
file_out = open("predictions-taskA.txt", "w")
for i in range(len(mlp_predict)):
    if i > 0:
        label = mlp_predict[i]
        # print(test_data["raw_data"][i])
        if label > 0.5:
            label = 1
        else:
            label = 0
        file_out.write("%d\n" % label)

file_out.close()
mlp_f1_scores = np.array(mlp_f1_scores)
print("Final mlp F1: %0.4f (+/- %0.4f)" % (mlp_f1_scores.mean(), mlp_f1_scores.std() * 2))
