# author: Wenrui Zhang
# email: wenruizhang@ucsb.edu
# 
# install required package using: pip install -r requirements.txt
# run the code: python main.py
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sklearn.model_selection as model_s
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

classes = ['Bulbasaur', 'Sudowoodo', 'Charmander', 'Gastly', 'Jigglypuff', 'Pidgey', 'Pikachu', 'Squirtle']
short_classes = ['Bul', 'Sud', 'Cha', 'Gas', 'Jig', 'Pid', 'Pik', 'Squ']

def preprocessing(data, labels=None): # preprocess data
    global classes
    samples = []
    # change the value of gender from char to float
    for sample in data:
        tmp = list(sample)
        tmp[9] = float(0) if sample[9] == 'F' else float(1)
        tmp = list(map(float, tmp))
        tmp = np.asarray(tmp, dtype=np.float32)
        samples.append(tmp)
    # input normalization
    scaler = StandardScaler()
    samples = scaler.fit_transform(samples)
    if labels is not None: # for training samples
        # encode the labels from string to 0-7
        le = LabelEncoder()
        le.fit(classes)
        labels = le.transform(labels)
        # split training data into training set and validtion set
        train_x, validate_x, train_y, validate_y = model_s.train_test_split(samples, labels, test_size=0.2, random_state=100)
        return train_x, validate_x, train_y, validate_y
    else: # for testing samples
        return samples


def k_nearest_neighbor(train_x, train_y, validate_x, validate_y):
    max_score = 0
    best_neighbors = None
    best_weights = None
    # grid search some possible combinations of hyper parameters
    for n_neighbors in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        for weights in ['uniform', 'distance']:
            # create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(train_x, train_y)
            score = clf.score(validate_x, validate_y)
            # record the best result
            max_score = score if score > max_score else max_score
            best_neighbors = n_neighbors if score == max_score else best_neighbors
            best_weights = weights if score == max_score else best_weights
            print("number of neighbors: ", n_neighbors, ", weights: ", weights)
            print(score)
    print("final result of KNN")
    print("number of neighbors: ", best_neighbors, ", weights: ", best_weights)
    print(max_score)
    # plot confusion matrix
    clf = neighbors.KNeighborsClassifier(best_neighbors, weights=best_weights)
    clf.fit(train_x, train_y)
    disp = plot_confusion_matrix(clf, validate_x, validate_y,
                                 display_labels=short_classes,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation='vertical')
    disp.ax_.set_title("Confusion Matrix of k-Nearest Neighbors")
    plt.savefig("CM_KNN.png")


def naive_bayes(train_x, train_y, validate_x, validate_y):
    # create an instance of Naive Bayes Classifier and fit the data.
    clf = GaussianNB()
    clf.fit(train_x, train_y)
    score = clf.score(validate_x, validate_y)
    print("result of Gaussian Naive Bayes")
    print(score)
    # plot confusion matrix
    disp = plot_confusion_matrix(clf, validate_x, validate_y,
                                 display_labels=short_classes,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation='vertical')
    disp.ax_.set_title("Confusion Matrix of Gaussian Naive Bayes")
    plt.savefig("CM_GNB.png")


def svm(train_x, train_y, validate_x, validate_y):
    parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]}
    # create an instance of SVM Classifier and fit the data.
    clf = SVC()
    # grid search some possible combinations of hyper parameters
    clf = model_s.GridSearchCV(clf, parameters)
    clf.fit(train_x, train_y)
    score = clf.score(validate_x, validate_y)
    print("result of SVM")
    print(score)
    print(clf.best_params_) # the hyper parameteres with best result.
    # plot confusion matrix
    disp = plot_confusion_matrix(clf, validate_x, validate_y,
                                 display_labels=short_classes,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation='vertical')
    disp.ax_.set_title("Confusion Matrix of SVM")
    plt.savefig("CM_SVM.png")


def decision_tree(train_x, train_y, validate_x, validate_y):
    # create an instance of DT Classifier and fit the data.
    clf = DecisionTreeClassifier()
    clf.fit(train_x, train_y)
    score = clf.score(validate_x, validate_y)
    print("result of decision tree")
    print(score)
    # plot confusion matrix
    disp = plot_confusion_matrix(clf, validate_x, validate_y,
                                 display_labels=short_classes,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation='vertical')
    disp.ax_.set_title("Confusion Matrix of decision tree")
    plt.savefig("CM_DT.png")


def lda(train_x, train_y, validate_x, validate_y):
    # create an instance of LDA Classifier and fit the data.
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_x, train_y)
    score = clf.score(validate_x, validate_y)
    print("result of LDA")
    print(score)
    # plot confusion matrix
    disp = plot_confusion_matrix(clf, validate_x, validate_y,
                                 display_labels=short_classes,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation='vertical')
    disp.ax_.set_title("Confusion Matrix of LDA")
    plt.savefig("CM_LDA.png")


def random_forest(train_x, train_y, validate_x, validate_y):
    # create an instance of RF Classifier and fit the data.
    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)
    score = clf.score(validate_x, validate_y)
    print("result of random forest")
    print(score)
    # plot confusion matrix
    disp = plot_confusion_matrix(clf, validate_x, validate_y,
                                 display_labels=short_classes,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation='vertical')
    disp.ax_.set_title("Confusion Matrix of random forest")
    plt.savefig("CM_RF.png")


def mlp(train_x, train_y, validate_x, validate_y):
    # test on different network sizes.
    hidden_size = [(400,), (800,), (1200,), (1600,), (2000,), (200, 200,), (400, 400,), (800, 800,), (400, 400, 400,),
                   (400, 400, 400, 400,)]
    max_accuracy = 0
    best_hidden = None
    for hidden in hidden_size:
        # create an instance of MLP Classifier and fit the data.
        clf = MLPClassifier(hidden_layer_sizes=hidden)
        clf.fit(train_x, train_y)
        score = clf.score(validate_x, validate_y)
        best_hidden = hidden if max_accuracy < score else best_hidden
        max_accuracy = score if max_accuracy < score else max_accuracy
        print("network size: ", hidden)
        print(score)
    print("result of MLP")
    print("network size: ", best_hidden)
    print(max_accuracy)
    # best result is reported
    # plot confusion matrix
    clf = MLPClassifier(hidden_layer_sizes=best_hidden)
    clf.fit(train_x, train_y)
    disp = plot_confusion_matrix(clf, validate_x, validate_y,
                                 display_labels=short_classes,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation='vertical')
    disp.ax_.set_title("Confusion Matrix of MLP")
    plt.savefig("CM_MLP.png")
    return best_hidden


def mlp_predict(train_x, train_y, test_x, best_hidden):
    # train all the training data using MLP with best network size and then predict the testing data
    global classes
    clf = MLPClassifier(hidden_layer_sizes=best_hidden)
    clf.fit(train_x, train_y)
    test_y = clf.predict(test_x)
    le = LabelEncoder()
    le.fit(classes)
    # decode the predicted labels to the name of Pokemons.
    predict_y = le.inverse_transform(test_y)
    np.save("pokemon_test_y.npy", predict_y)
    print(predict_y)


# load data
train_x = np.load("pokemon_train_x.npy")
train_y = np.load("pokemon_train_y.npy")
test_x = np.load("pokemon_test_x.npy")

train_x, validate_x, train_y, validate_y = preprocessing(train_x, train_y)

k_nearest_neighbor(train_x, train_y, validate_x, validate_y)
naive_bayes(train_x, train_y, validate_x, validate_y)
svm(train_x, train_y, validate_x, validate_y)
decision_tree(train_x, train_y, validate_x, validate_y)
lda(train_x, train_y, validate_x, validate_y)
random_forest(train_x, train_y, validate_x, validate_y)
best_hidden = mlp(train_x, train_y, validate_x, validate_y)

x = np.concatenate((train_x, validate_x), axis=0)
y = np.concatenate((train_y, validate_y), axis=0)
test_x = preprocessing(test_x)
mlp_predict(x, y, test_x, best_hidden)
