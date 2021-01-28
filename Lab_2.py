import numpy as np
import string
from collections import Counter
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score

np.random.seed(1)

# Problem 1

def count_frequency(documents):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    lower_case_doc = []
    for s in documents:
        lower_case_doc.append(str(s.lower()))

    no_punc_doc = []
    for s in lower_case_doc:
        for x in s:
            if x in punctuations:
                s = s.replace(x, "")
        no_punc_doc.append(s)

    word_doc = []
    for s in no_punc_doc:
        word_doc = word_doc + s.split()

    frequency = Counter()
    for s in word_doc:
        frequency[s] += 1
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return frequency

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
def prior_prob(y_train):
    prior = {}
    length = len(y_train)
    unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    #     print(unique_elements, counts_elements)
    prior = {i: (j / length) for i, j in zip(unique_elements, counts_elements)}

    return prior


def conditional_prob(X_train, y_train):
    cond_prob = {}
    N = 20000
    ALPHA = 1
    unique_elements = np.unique(y_train)
    posZero = X_train[np.where(y_train == 0)]
    posOne = X_train[np.where(y_train == 1)]

    notSpam = count_frequency(posZero)
    countY = np.sum(list(notSpam.values()))

    spam = count_frequency(posOne)
    countX = np.sum(list(spam.values()))

    cond_prob = {}
    dictNS = {i: (j + ALPHA) / ((N) * (ALPHA) + countY) for i, j in notSpam.items()}
    dictS = {i: (j + ALPHA) / (N * (ALPHA) + countX) for i, j in spam.items()}

    dict_list = [dictNS, dictS]
    for label, value in zip(unique_elements, dict_list):
        cond_prob[label] = value


    return cond_prob

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def predict_label(X_test, prior_prob, cond_prob):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = 20000
    ALPHA = 1
    predict = []
    val = (ALPHA) / (N * ALPHA)
    test_prob = []

    for sample in X_test:
        test_text = count_frequency([sample])
        list_Zero = []
        list_one = []
        for key in test_text:
            if (key in cond_prob[0]):
                list_Zero.append(cond_prob[0].get(key))
            else:
                list_Zero.append(val)

            if (key in cond_prob[1]):
                list_one.append(cond_prob[1].get(key))
            else:
                list_one.append(val)
        gofxZero = np.log(prior_prob[0]) + np.sum(np.log(list_Zero))

        gofxOne = np.log(prior_prob[1]) + np.sum(np.log(list_one))
        softmax_list = list(softmax([gofxZero, gofxOne]))
        test_prob.append(softmax_list)
        predict.append(np.argmax(softmax_list))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return predict, test_prob

def compute_test_prob(word_count, prior_cat, cond_cat):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    prob = 0
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return prob

def compute_metrics(y_pred, y_true):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    acc = accuracy_score(y_true, y_pred) * 100

    cm = confusion_matrix(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return acc, cm, f1


# Problem 2

def featureNormalization(X):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized, X_mean, X_std


def applyNormalization(X, X_mean, X_std):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X_normalized = (X - X_mean) / X_std

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized

def computeMSE(X, y, theta):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  total_samples, num_features = X.shape

  f_x = np.multiply(np.transpose(theta), X)
  f_x = np.sum(f_x, axis=1) - y
  l = np.square(f_x)
  p = 2 * total_samples
  error = (np.sum(l / p))

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return error[0]

def computeGradient(X, y, theta):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  f_x = np.dot(X.T, X)
  f_x = np.dot(f_x, theta)
  g_x = np.dot(y, X)
  #   g_x=np.sum(np.multiply(y,), axis=0)
  n, m = X.shape
  #   print(np.transpose(f_x) - g_x)
  gradient = (1 / n) * np.transpose(np.transpose(f_x) - g_x)

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  return gradient

def gradientDescent(X, y, theta, alpha, num_iters):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  Loss_record = np.zeros(num_iters)
  for i in range(0, num_iters):
      # theta = theta - alpha*(1.0/m) * np.transpose(X).dot(X.dot(theta) - np.transpose([y]))
      theta = theta - np.multiply(alpha, computeGradient(X, y, theta))
      Loss_record[i] = computeMSE(X, y, theta)

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return theta, Loss_record

def closeForm(X, y):

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    first = np.dot(X.T, X)
    second = np.dot(X.T, y.T)
    theta = np.dot(np.linalg.inv(first), second)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return np.expand_dims(theta, axis=1)
