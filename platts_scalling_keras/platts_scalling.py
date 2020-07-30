from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from platts_scalling_keras.network import create_model, get_gens
import numpy as np
import pandas as pd
import ast

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def generator_true_classes_to_hot_one(true_classes, num_classes):
    """

    :param true_classes: list of lists in ridicilous form of: [[3, 2], [0, 2], [3, 2], [1]] etc
    :return: a list of results in 1-hot encoding (but multilabeled)
    """
    results = np.zeros(shape=(len(true_classes), num_classes))
    for index, labels in enumerate(true_classes):
        for label_index in labels:
            results[index][label_index] = 1

    return results

#https://datascience.stackexchange.com/questions/41426/probability-calibration-role-of-hidden-layer-in-neural-network

def probability_scalling(classifier, X_train, y_train, X_test, y_test):
    # https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#sphx-glr-auto-examples-calibration-plot-calibration-py

    clf_sigmoid = CalibratedClassifierCV(classifier, cv=2, method='sigmoid')
    clf_sigmoid.fit(X_train, y_train)
    prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

    clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid)
    print("With calibration: %1.3f (smaller the better)" % clf_sigmoid_score)
    debug = 5

    return clf_sigmoid


# Function to create model, required for KerasClassifier



test_df = pd.read_csv('/mnt/efs/classification_csv/shuffled_valid_with_binary.csv',converters={1:ast.literal_eval})
train_df = pd.read_csv('/mnt/efs/classification_csv/shuffled_test_with_binary.csv',converters={1:ast.literal_eval})
path_to_trained_model = r'/mnt/efs/classification_results/algorithm_results/model_2020-07-23-22-04-35_MobileNetV2 - fine tuned/best_model.h5'

train_gen, test_gen = get_gens(test_df, train_df)

steps_per_epoch = np.ceil(train_gen.samples / train_gen.batch_size)

model = create_model(path_to_trained_model)

test_steps_per_epoch = np.ceil(train_gen.samples / train_gen.batch_size)

predictions = model.predict_generator(train_gen, steps=test_steps_per_epoch)

true_classes = train_df['BINARY_NUDE'].values
true_classes = true_classes * 1 # bool to int
debug = 5

X_train, X_test, y_train, y_test,  =  train_test_split(predictions, true_classes, test_size=0.3, random_state=42, stratify=true_classes)


#https://datascience.stackexchange.com/questions/41426/probability-calibration-role-of-hidden-layer-in-neural-network

#ir = IsotonicRegression()
#ir = LogisticRegression(C=1e5)
ir = RandomForestClassifier()
ir.fit(X_train,y_train)
iso_presictions = ir.predict(X_test)

cm = confusion_matrix(y_test, iso_presictions)
f = sns.heatmap(cm, annot=True)
plt.show()

#last time results
#array([[2257,   14],
       #[  24,  239]])


f1_res = f1_score(y_test, iso_presictions)
print('f1:',f1_res)
report = classification_report(y_test, iso_presictions)
print(report)

debug =5