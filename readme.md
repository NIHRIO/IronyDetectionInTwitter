
# A Simple and Accurate Neural Network Model for Irony Detection in Twitter

This program provides the implementation of our model for the SemEval 2018 Task 3: Irony Detection in English Tweets, as described in our paper:

    @InProceedings{VUN2018,
          author    = {Vu, Thanh and Nguyen, Dat Quoc and Vu, Xuan-Son and Nguyen, Dai Quoc and Catt, Michael and Trenell, Michael},
          title     = {{NIHRIO at SemEval-2018 Task 3: A Simple and Accurate Neural Network Model for Irony Detection in Twitter}},
          booktitle = {Proceedings of the 12th International Workshop on Semantic Evaluation},
          year      = {2018}
    }

> **Brief description:** In this paper, we propose to use a simple neural network architecture of Multilayer Perceptron with various types of input features including lexical, syntactic, semantic and polarity features (***see the following figure***). Our system achieves very high performance in both subtasks of binary and multi-class irony detection in tweets. In particular, we rank at least **<u>third** using the accuracy metric and **<u>fifth** using the F1 metric. 

<p align="center">
    <img src="https://github.com/NIHRIO/IronyDetectionInTwitter/blob/master/description/mlp.png" alt="Overview of our model architecture" width="50%"/>
</p>

Please cite the paper above when the model is used to produce published results or incorporated into other software. 

We would highly appreciate having your bug reports, comments and suggestions. As a free open-source implementation, the implementation is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

## Setup and Reproduce the results
To run the code, you need the following software packages by running **`pip install -r requirements.txt`**:

- `python 2.7`
- `emoji 0.4.5`
- `scikit-learn 0.19.1`
- `spacy 2.0.9`
- `nltk 3.2.5`
- `tensorflow 1.4`

One the packages are installed, you can run the code as follows:
- `python src/run_task_A.py` for the subtask A of binary irony (ironic vs. non-ironic) classification
- `python src/run_task_B.py` for the subtask B of different types of irony classification

For each command, it will produce a file for the corresponding subtask, which consists of the predicted labels for the test data. 

## Customization 
You can also customize the model to other classification tasks using the code provided in `src/MLP.py`
