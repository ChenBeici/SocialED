# -*- coding: utf-8 -*-
"""Evaluation metrics for clustering and classification in social event detection."""

import numpy as np
import torch
from sklearn import metrics
from munkres import Munkres
import networkx as nx



def eval_nmi(ground_truths, predictions):
    """
    Normalized Mutual Information (NMI) score for clustering evaluation.

    Parameters
    ----------
    ground_truths : array-like
        Ground truth labels.
    predictions : array-like 
        Predicted cluster labels.

    Returns
    -------
    nmi : float
        Normalized Mutual Information score.
    """
    nmi = metrics.normalized_mutual_info_score(ground_truths, predictions)
    return nmi


def eval_ami(ground_truths, predictions):
    """
    Adjusted Mutual Information (AMI) score for clustering evaluation.

    Parameters
    ----------
    ground_truths : array-like
        Ground truth labels.
    predictions : array-like
        Predicted cluster labels.

    Returns
    -------
    ami : float
        Adjusted Mutual Information score.
    """
    ami = metrics.adjusted_mutual_info_score(ground_truths, predictions)
    return ami


def eval_ari(ground_truths, predictions):
    """
    Adjusted Rand Index (ARI) score for clustering evaluation.

    Parameters
    ----------
    ground_truths : array-like
        Ground truth labels.
    predictions : array-like
        Predicted cluster labels.

    Returns
    -------
    ari : float
        Adjusted Rand Index score.
    """
    ari = metrics.adjusted_rand_score(ground_truths, predictions)
    return ari


def eval_f1(ground_truths, predictions):
    """
    F1 score for classification evaluation.

    Parameters
    ----------
    ground_truths : array-like
        Ground truth labels.
    predictions : array-like
        Predicted labels.

    Returns
    -------
    f1 : float
        F1 score.
    """
    f1 = metrics.f1_score(ground_truths, predictions, average='macro')
    return f1


def eval_acc(ground_truths, predictions):
    """
    Accuracy score for classification evaluation.

    Parameters
    ----------
    ground_truths : array-like
        Ground truth labels.
    predictions : array-like
        Predicted labels.

    Returns
    -------
    acc : float
        Accuracy score.
    """
    acc = metrics.accuracy_score(ground_truths, predictions)
    return acc

