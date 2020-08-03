import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

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



# Function to create model, required for KerasClassifier

def prepare():

    test_df = pd.read_csv('/mnt/efs/classification_csv/shuffled_valid_with_binary.csv',converters={1:ast.literal_eval})
    train_df = pd.read_csv('/mnt/efs/classification_csv/shuffled_test_with_binary.csv',converters={1:ast.literal_eval})
    path_to_trained_model = r'/mnt/efs/classification_results/algorithm_results/model_2020-07-23-22-04-35_MobileNetV2 - fine tuned/best_model.h5'

    train_gen, test_gen = get_gens(test_df, train_df)

    model = create_model(path_to_trained_model)

    train_steps_per_epoch = np.ceil(train_gen.samples / train_gen.batch_size)
    test_steps_per_epoch = np.ceil(test_gen.samples / test_gen.batch_size)

    model_train_predictions = model.predict_generator(train_gen, steps=train_steps_per_epoch)

    train_true_classes = train_df['BINARY_NUDE'].values
    train_true_classes = train_true_classes * 1 # bool to int

    # X_train, X_test, y_train, y_test,  =  train_test_split(predictions, true_classes, test_size=0.3, random_state=42, stratify=true_classes)
    X_train = model_train_predictions
    y_train = train_true_classes

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)

    model_test_predictions = model.predict_generator(test_gen, steps=test_steps_per_epoch)
    test_true_classes = test_df['BINARY_NUDE'].values

    X_test = model_test_predictions
    y_test = test_true_classes

    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    return X_train,y_train,X_test,y_test

def test(X_train,y_train,X_test,y_test):
    num_folds = 10
    seed = 7
    scoring = 'f1_macro'


    models = []
    models.append(('LR', LogisticRegression(solver='liblinear')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    models.append(('AB', AdaBoostClassifier()))
    models.append(('GBM', GradientBoostingClassifier()))
    models.append(('RF', RandomForestClassifier(n_estimators=10)))
    models.append(('ET', ExtraTreesClassifier(n_estimators=10)))
    results = []
    names = []

    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    pyplot.show()


    #tuning best model
    neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    param_grid = dict(n_neighbors=neighbors)
    model = KNeighborsClassifier()

    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, iid=True)
    grid_result = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))




    #finilizing model
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)

    print(confusion_matrix(y_test, test_predictions))
    print(classification_report(y_test, test_predictions))

    filename = 'finalized_model.sav'
    joblib.dump(model, filename)



if __name__ == '__main__':


    #X_train, y_train, X_test, y_test = prepare()

    print('MAKING DATA UP')
    X1, Y1 = make_classification(n_features=9, n_redundant=0, n_informative=5, n_classes=2, n_samples=10000)
    X_train, X_test, y_train, y_test,  =  train_test_split(X1, Y1, test_size=0.3, random_state=42, stratify=Y1)

    test(X_train,y_train,X_test,y_test)


#https://datascience.stackexchange.com/questions/41426/probability-calibration-role-of-hidden-layer-in-neural-network

# #ir = IsotonicRegression()
# #ir = LogisticRegression(C=1e5)
# ir = RandomForestClassifier()
# ir.fit(X_train,y_train)
# iso_presictions = ir.predict(X_test)
#
# cm = confusion_matrix(y_test, iso_presictions)
# f = sns.heatmap(cm, annot=True)
# plt.show()
#
# #last time results
# #array([[2257,   14],
#        #[  24,  239]])
#
#
# f1_res = f1_score(y_test, iso_presictions)
# print('f1:',f1_res)
# report = classification_report(y_test, iso_presictions)
# print(report)

debug =5