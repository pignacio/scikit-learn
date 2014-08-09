"""
The :mod:`sklearn.model_selection` module includes utilities for selecting a
model and tuning its parameters .
"""

from .partition import KFold, LeaveOneLabelOut, LeaveOneOut, LeavePLabelOut
from .partition import LeavePOut, ShuffleSplit, StratifiedKFold
from .partition import StratifiedShuffleSplit, check_cv, train_test_split
from .validate import cross_val_score, permutation_test_score

__all__ = ['KFold',
           'LeaveOneLabelOut',
           'LeaveOneOut',
           'LeavePLabelOut',
           'LeavePOut',
           'ShuffleSplit',
           'StratifiedKFold',
           'StratifiedShuffleSplit',
           'check_cv',
           'cross_val_score',
           'permutation_test_score',
           'train_test_split']
