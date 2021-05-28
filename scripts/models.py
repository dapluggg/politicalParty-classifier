#%%
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import numpy as np
import pprint as pprint
import math
import pandas as pd
from sklearn.metrics import roc_curve
from matplotlib import pyplot

#%%
def train_multinomNB(features, labels, classes, folds=5):
    """Train MultinomialNB() model on a set of features, given labels.

    Args:
        features (array): Array of feature counts. Array may be TF-IDF weighted. 
        labels (array): 1-D Array of true labels. 
        classes (list): List of possible classes. The first class is assumed to be the positive class. 

    Returns:
        None: All output printed to console. 
    """

    kf = KFold(n_splits=folds, shuffle=True, random_state=999)
    kf.get_n_splits(features)
    f1scores = []
    accuracies = []
    confusions = [[0,0], [0,0]]
    log_ratios = []
    
    for train_indices, test_indices in kf.split(features):
        train_x = features.iloc[train_indices]
        train_y = labels.iloc[train_indices]
        test_x = features.iloc[test_indices]
        test_y = labels.iloc[test_indices]

        naive_bayes_classifier = MultinomialNB()
        naive_bayes_classifier.fit(train_x, train_y)

        predictions = naive_bayes_classifier.predict(test_x)
        lr_probs = naive_bayes_classifier.predict_proba(test_x)
        lr_probs = lr_probs[:, 0]        
        confusions += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label=classes[0])
        f1scores.append(score)
        accuracy = accuracy_score(test_y, predictions)
        accuracies.append(accuracy)


        pos_scores = naive_bayes_classifier.feature_log_prob_[0, :]
        neg_scores = naive_bayes_classifier.feature_log_prob_[1, :]
        ratio = neg_scores/pos_scores
        log_ratio = [math.log(item) for item in ratio]
        log_ratios.append(log_ratio)

        # How the HELL do you get the top n features...
        # I keep getting math domain error 
        # is it beause log(0) is undefined? 

        # Figured it out, had to do log(x/y) instead of log(x) - log(y)
        # Mathematically this is the same but idk why this works 
        # too tired to figure out why...
        


        # Plotting AUC ROC 
        fpr, tpr, threshhold = roc_curve(test_y, lr_probs, pos_label=classes[0])
        pyplot.plot(fpr, tpr, marker='.', label='Logistic')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()   
    print('Important words (Democratic):')
    log_ratios = np.array(log_ratios)
    top_features = {
        'feature': features.columns,
        'score': log_ratios.mean(axis=0)
    }
    top_features = pd.DataFrame(top_features).sort_values('score', ascending=False)
    print(top_features['feature'].head(n=20))
    print('\n')
    print('Important words (Republican):')
    print(top_features['feature'].tail(n=20))
    print('\n')
    print(f'Average F1-Score: {np.mean(f1scores)}')
    print('\n')
    print(f'Average Accuracy: {np.mean(accuracies)}')
    pprint.pprint(confusions)
    print('\n\n\n')


    return None

def train_bernoulliNB(features, labels, classes, folds=5):
    """Train MultinomialNB() model on a set of features, given labels.

    Args:
        features (array): Array of feature counts. Array may be TF-IDF weighted. 
        labels (array): 1-D Array of true labels. 
        classes (list): List of possible classes. The first class is assumed to be the positive class. 

    Returns:
        None: All output printed to console. 
    """

    kf = KFold(n_splits=folds, shuffle=True, random_state=999)
    kf.get_n_splits(features)
    f1scores = []
    accuracies = []
    confusions = [[0,0], [0,0]]
    log_ratios = []
    
    for train_indices, test_indices in kf.split(features):
        train_x = features.iloc[train_indices]
        train_y = labels.iloc[train_indices]
        test_x = features.iloc[test_indices]
        test_y = labels.iloc[test_indices]

        naive_bayes_classifier = BernoulliNB()
        naive_bayes_classifier.fit(train_x, train_y)

        predictions = naive_bayes_classifier.predict(test_x)
        lr_probs = naive_bayes_classifier.predict_proba(test_x)
        lr_probs = lr_probs[:, 1]
        confusions += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label=classes[0])
        f1scores.append(score)
        accuracy = accuracy_score(test_y, predictions)
        accuracies.append(accuracy)


        pos_scores = naive_bayes_classifier.feature_log_prob_[0, :]
        neg_scores = naive_bayes_classifier.feature_log_prob_[1, :]
        ratio = neg_scores/pos_scores
        log_ratio = [math.log(item) for item in ratio]
        log_ratios.append(log_ratio)

        # How the HELL do you get the top n features...
        # I keep getting math domain error 
        # is it beause log(0) is undefined? 

        # Figured it out, had to do log(x/y) instead of log(x) - log(y)
        # Mathematically this is the same but idk why this works 
        # too tired to figure out why...
        


        # Plotting AUC ROC 
        fpr, tpr, threshhold = roc_curve(test_y, lr_probs, pos_label=classes[0])
        pyplot.plot(fpr, tpr, marker='.', label='Logistic')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()   
    print('Important words (Democratic):')
    log_ratios = np.array(log_ratios)
    top_features = {
        'feature': features.columns,
        'score': log_ratios.mean(axis=0)
    }
    top_features = pd.DataFrame(top_features).sort_values('score', ascending=False)
    print(top_features.columns)
    # print(top_features['feature'].head(n=20))
    # print('\n')
    # print('Important words (Republican):')
    # print(top_features['feature'].tail(n=20))
    # print('\n')
    # print(f'Average F1-Score: {np.mean(f1scores)}')
    # print('\n')
    # print(f'Average Accuracy: {np.mean(accuracies)}')
    # pprint.pprint(confusions)
    # print('\n\n\n')


    return None

