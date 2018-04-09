"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======
"""

from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays

LFW_PEOPLE = fetch_lfw_people(
    min_faces_per_person=70, resize=0.4, data_home='./scikit_learn_data')

# introspect the images arrays to find the shapes (for plotting)
N_SAMPLES, HEIGHT, WIDTH = LFW_PEOPLE.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = LFW_PEOPLE.data
N_FEATURES = X.shape[1]

# the label to predict is the id of the person
Y = LFW_PEOPLE.target
TARGET_NAMES = LFW_PEOPLE.target_names
N_CLASSES = TARGET_NAMES.shape[0]

print("Total dataset size:")
print("n_samples: %d" % N_SAMPLES)
print("n_features: %d" % N_FEATURES)
print("n_classes: %d" % N_CLASSES)

# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    X, Y, test_size=0.25, random_state=42)

# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
N_COMPONENTS = 150

print("Extracting the top %d eigenfaces from %d faces" % (N_COMPONENTS,
                                                          X_TRAIN.shape[0]))
T0 = time()
PCA = PCA(
    n_components=N_COMPONENTS, svd_solver='randomized',
    whiten=True).fit(X_TRAIN)
print("done in %0.3fs" % (time() - T0))

EIGENFACES = PCA.components_.reshape((N_COMPONENTS, HEIGHT, WIDTH))

print("Projecting the input data on the eigenfaces orthonormal basis")
T0 = time()
X_TRAIN_PCA = PCA.transform(X_TRAIN)
X_TEST_PCA = PCA.transform(X_TEST)
print("done in %0.3fs" % (time() - T0))

# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
T0 = time()
PARAM_GRID = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}
CLF = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), PARAM_GRID)
CLF = CLF.fit(X_TRAIN_PCA, Y_TRAIN)
print("done in %0.3fs" % (time() - T0))
print("Best estimator found by grid search:")
print(CLF.best_estimator_)

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
T0 = time()
Y_PRED = CLF.predict(X_TEST_PCA)
print("done in %0.3fs" % (time() - T0))

print(classification_report(Y_TEST, Y_PRED, target_names=TARGET_NAMES))
print(confusion_matrix(Y_TEST, Y_PRED, labels=range(N_CLASSES)))

# #############################################################################
# Qualitative evaluation of the predictions using matplotlib


def plot_gallery(images, titles, height, width, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((height, width)), cmap='gray')
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set


def title(y_pred, y_test, target_names, i):
    """Helper function to plot the result of the prediction"""
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


PREDICTION_TITLES = [
    title(Y_PRED, Y_TEST, TARGET_NAMES, i) for i in range(Y_PRED.shape[0])
]

plot_gallery(X_TEST, PREDICTION_TITLES, HEIGHT, WIDTH)

# plot the gallery of the most significative eigenfaces

EIGENFACE_TITLES = ["eigenface %d" % i for i in range(EIGENFACES.shape[0])]
plot_gallery(EIGENFACES, EIGENFACE_TITLES, HEIGHT, WIDTH)

plt.show()
