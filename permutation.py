from sklearn.utils import (indexable, check_random_state, _safe_indexing)
from sklearn.model_selection._split import check_cv
from sklearn.metrics import check_scoring
from sklearn.base import is_classifier, clone
from sklearn.utils.metaestimators import _safe_split
import numpy as np
from models import CorrMatrix

def permutations(estimator, X, y, cv=None, n_permuations=100, random_state=0, scoring=None):
    """
    This follows the sklearn API sklearn.inspection.permutation_test_score
    I have modified accordinlgy to accomodate filtering of features using correlation matrix
    before running cross-validation using the model
    """

    Xs, ys = indexable(X, y)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    random_state = check_random_state(random_state)

    # corr = CorrMatrix()
    # corr.fit(X,y)
    # Xs, ys = corr.transform()
    score = _permutations(clone(estimator), Xs, ys, cv, scorer)
    permutation_scores = np.zeros((n_permuations))
    for i in range(n_permuations):
        corr_p = CorrMatrix()
        corr_p.fit(X, y)
        Xp, yp = corr_p.transform()
        yp = _safe_indexing(yp, random_state.permutation(len(yp)))
        permutation_scores[i] = _permutations(clone(estimator), Xp, yp, cv, scorer)

    pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permuations + 1)

    return score, permutation_scores, pvalue

def _permutations(estimator, X, y, cv, scorer):
    """Auxiliary function for permutations"""
    avg_score = []
    for train, test in cv.split(X,y):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        estimator.fit(X_train, y_train)
        avg_score.append(scorer(estimator, X_test, y_test))
    return np.mean(avg_score)

