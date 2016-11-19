from __future__ import division

'''
This file contains implementation of derivatives of loss functions.
'''


def SSE(true, pred):
    return pred - true


def cross_entropy(true, pred):
    return (pred - true) / (pred * (1 - pred))
