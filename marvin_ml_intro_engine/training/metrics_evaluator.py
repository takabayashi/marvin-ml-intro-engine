#!/usr/bin/env python
# coding=utf-8

"""MetricsEvaluator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['MetricsEvaluator']


logger = get_logger('metrics_evaluator')


class MetricsEvaluator(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(MetricsEvaluator, self).__init__(**kwargs)

    def execute(self, **kwargs):
        from sklearn.metrics import accuracy_score
        from time import time

        t0 = time()
        y_pred = self.model.predict(self.dataset["features_test_transformed"])
        print "prediction time:", round(time() - t0, 3), "s"

        accuracy_score = accuracy_score(self.dataset["labels_test"], y_pred)

        print "the accuracy score is ", accuracy_score
        self.metrics = {"accuracy_score": accuracy_score}

