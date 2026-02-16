# %% [markdown]

# # Programming assignment 4: Implementing linear classifiers

# ## Task 1: Example of linearly non-separable data

# %% [python]
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

X1 = [
    {"city": "Gothenburg", "month": "July"},
    {"city": "Gothenburg", "month": "December"},
    {"city": "Paris", "month": "July"},
    {"city": "Paris", "month": "December"},
]
Y1 = ["rain", "rain", "sun", "rain"]

X2 = [
    {"city": "Sydney", "month": "July"},
    {"city": "Sydney", "month": "December"},
    {"city": "Paris", "month": "July"},
    {"city": "Paris", "month": "December"},
]
Y2 = ["rain", "sun", "sun", "rain"]

classifier1 = make_pipeline(DictVectorizer(), Perceptron(max_iter=10))
classifier1.fit(X1, Y1)
guesses1 = classifier1.predict(X1)
print(accuracy_score(Y1, guesses1))

# classifier2 = make_pipeline(DictVectorizer(), LinearSVC())
classifier2 = make_pipeline(DictVectorizer(), Perceptron(max_iter=10))
classifier2.fit(X2, Y2)
guesses2 = classifier2.predict(X2)
print(accuracy_score(Y2, guesses2))

# %% [markdown]

# The first dataset is linearly separable, so the Perceptron is able to classify with 100% accuracy.
# The second dataset is not linearly separable, so the Perceptron is unable to classify with more than 75% accuracy.
# We only get 50% accuracy due to the probability of the perceptron choosing a hyperplane that correctly classifies the first three examples, but misclassifies the last one.
# If we were to run the code enough times, we would see that the accuracy varies between 50% and 75%.

# %% [markdown]
# ## Task 2: Preparation for next tasks

# In order to make the code run properly we needed to update some things.
# 1. We needed to have the `fit` method of the `Perceptron` class return `self`, so that it can be used in a pipeline.
# 2. We needed to update variable names in the `LinearClassifier` class to have a trailing underscore, which is a common convention in scikit-learn for variables that are set during fitting.
# 3. We needed to update the `predict` method of the `LinearClassifier` class to use `np.where` instead of `np.select`, which is more appropriate for this use.
#    This was not strictly necessary, but otherwise we had to use a default value in `np.select`.
#
# After this, the training time was approx. 1s, and the accuracy was 0.7919.

# %% [markdown]
# ## Task 3: Implementing the SVC

# %% [python]
import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


class LinearClassifier(BaseEstimator):
    """
    General class for binary linear classifiers. Implements the predict
    function, which is the same for all binary linear classifiers. There are
    also two utility functions.
    """

    def decision_function(self, X):
        """
        Computes the decision function for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """
        return X.dot(self.w_)

    def predict(self, X):
        """
        Predicts the outputs for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """

        # First compute the output scores
        scores = self.decision_function(X)

        # Select the positive or negative class label, depending on whether
        # the score was positive or negative.
        return np.where(scores >= 0.0, self.positive_class_, self.negative_class_)

    def find_classes(self, Y):
        """
        Finds the set of output classes in the output part Y of the training set.
        If there are exactly two classes, one of them is associated to positive
        classifier scores, the other one to negative scores. If the number of
        classes is not 2, an error is raised.
        """
        classes = sorted(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class_ = classes[1]
        self.negative_class_ = classes[0]

    def encode_outputs(self, Y):
        """
        A helper function that converts all outputs to +1 or -1.
        """
        return np.array([1 if y == self.positive_class_ else -1 for y in Y])


class Perceptron(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm.
    """

    def __init__(self, n_iter=20):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

    def fit(self, X, Y):
        self.find_classes(Y)
        Ye = self.encode_outputs(Y)

        if not isinstance(X, np.ndarray):
            X = X.toarray()

        n_samples, n_features = X.shape
        self.w_ = np.zeros(n_features)

        t = 0  # Global step counter
        lam = 1 / n_samples  # Regularization strength
        for epoch in range(1, self.n_iter + 1):
            for x, y in zip(X, Ye):
                t += 1
                eta = 1.0 / (lam * t)
                score = np.dot(x, self.w_)
                if y * score < 1:
                    # Update with both regularization and the gradient of the loss
                    self.w_ = (1 - eta * lam) * self.w_ + (eta * y * x)
                else:
                    # Update with only regularization
                    self.w_ = (1 - eta * lam) * self.w_

        return self


def read_data(corpus_file):
    X = []
    Y = []
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            _, y, _, x = line.split(maxsplit=3)
            X.append(x.strip())
            Y.append(y)
    return X, Y


X, Y = read_data("data/all_sentiment_shuffled.txt")

# Split into training and test parts.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

highest_score = -1
highest_iter = -1
for i in range(1, 40):
    # Set up the preprocessing steps and the classifier.
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),
        # NB that this is our Perceptron, not skPerceptronlearn.linear_model.Perceptron
        Perceptron(n_iter=i),
    )

    # Train the classifier.
    t0 = time.time()
    pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    print("Training time: {:.2f} sec.".format(t1 - t0))

    # Evaluate on the test set.
    Yguess = pipeline.predict(Xtest)
    print("Accuracy: {:.4f}.".format(accuracy_score(Ytest, Yguess)))
    if accuracy_score(Ytest, Yguess) > highest_score:
        highest_score = accuracy_score(Ytest, Yguess)
        highest_iter = i

print(f"Highest accuracy: {highest_score:.4f} with n_iter={highest_iter}")

# %% [markdown]

# The highest accuracy we got was 0.8326 with n_iter=15
# Additionally, when manually settings the lambda parameter to 1e-4, we got an accuracy of 0.8352 with n_iter=10.


# %% [markdown]
# ## Task 4: Logistic Regression

# %% [python]
import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


class LinearClassifier(BaseEstimator):
    """
    General class for binary linear classifiers. Implements the predict
    function, which is the same for all binary linear classifiers. There are
    also two utility functions.
    """

    def decision_function(self, X):
        """
        Computes the decision function for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """
        return X.dot(self.w_)

    def predict(self, X):
        """
        Predicts the outputs for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """

        # First compute the output scores
        scores = self.decision_function(X)

        # Select the positive or negative class label, depending on whether
        # the score was positive or negative.
        return np.where(scores >= 0.0, self.positive_class_, self.negative_class_)

    def find_classes(self, Y):
        """
        Finds the set of output classes in the output part Y of the training set.
        If there are exactly two classes, one of them is associated to positive
        classifier scores, the other one to negative scores. If the number of
        classes is not 2, an error is raised.
        """
        classes = sorted(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class_ = classes[1]
        self.negative_class_ = classes[0]

    def encode_outputs(self, Y):
        """
        A helper function that converts all outputs to +1 or -1.
        """
        return np.array([1 if y == self.positive_class_ else -1 for y in Y])


class Perceptron(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm.
    """

    def __init__(self, n_iter=20):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

    def fit(self, X, Y):
        self.find_classes(Y)
        Ye = self.encode_outputs(Y)

        if not isinstance(X, np.ndarray):
            X = X.toarray()

        n_samples, n_features = X.shape
        self.w_ = np.zeros(n_features)

        t = 0  # Global step counter
        lam = 1 / n_samples  # Regularization strength
        for epoch in range(1, self.n_iter + 1):
            for x, y in zip(X, Ye):
                t += 1
                eta = 1.0 / (lam * t)
                score = np.dot(x, self.w_)
                step = (eta * y * x) / (1.0 + np.exp(y * score))

                # Update: w_new = w_old + eta * step
                w_old = (1 - 1.0 / t) * self.w_
                self.w_ = w_old + step

        return self


def read_data(corpus_file):
    X = []
    Y = []
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            _, y, _, x = line.split(maxsplit=3)
            X.append(x.strip())
            Y.append(y)
    return X, Y


X, Y = read_data("data/all_sentiment_shuffled.txt")

# Split into training and test parts.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

highest_score = -1
highest_iter = -1
for i in range(1, 40):
    # Set up the preprocessing steps and the classifier.
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),
        # NB that this is our Perceptron, not skPerceptronlearn.linear_model.Perceptron
        Perceptron(n_iter=i),
    )

    # Train the classifier.
    t0 = time.time()
    pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    print("Training time: {:.2f} sec.".format(t1 - t0))

    # Evaluate on the test set.
    Yguess = pipeline.predict(Xtest)
    print("Accuracy: {:.4f}.".format(accuracy_score(Ytest, Yguess)))
    if accuracy_score(Ytest, Yguess) > highest_score:
        highest_score = accuracy_score(Ytest, Yguess)
        highest_iter = i

print(f"Highest accuracy: {highest_score:.4f} with n_iter={highest_iter}")

# %% [markdown]
# The highest accuracy we got was 0.8347 with n_iter=35
# In this case we see significant improvements with further iterations compared to the hinge loss.
