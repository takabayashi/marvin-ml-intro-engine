#!/usr/bin/env python
# coding=utf-8

"""Trainer engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['Trainer']


logger = get_logger('trainer')


class Trainer(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

    def execute(self, **kwargs):
        """ 
            This is the code to accompany the Lesson 3 (Decision Tree) mini-project. 

            Use a Decision Tree to identify emails by their authors

            authors and labels:
            Sara has label 0
            Chris has label 1
        """
        from sklearn.tree import DecisionTreeClassifier
        from time import time

        print "Starting traning process..."
        t0 = time()

        clf = DecisionTreeClassifier(min_samples_split=40)

        clf.fit(self.dataset["features_train_transformed"], self.dataset["labels_train"])

        print "training time:", round(time() - t0, 3), "s"

        self.model = clf

        print "Done!"

