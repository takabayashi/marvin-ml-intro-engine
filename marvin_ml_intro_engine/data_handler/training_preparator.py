#!/usr/bin/env python
# coding=utf-8

"""TrainingPreparator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['TrainingPreparator']


logger = get_logger('training_preparator')


class TrainingPreparator(EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(TrainingPreparator, self).__init__(**kwargs)

    def execute(self, **kwargs):
        from sklearn import cross_validation
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_selection import SelectPercentile, f_classif

        # test_size is the percentage of events assigned to the test set
        # (remainder go into training)
        features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
            self.initial_dataset["word_data"],
            self.initial_dataset["authors"],
            test_size=self.params["test_size"],
            random_state=self.params["random_state"])

        # text vectorization--go from strings to lists of numbers
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        features_train_transformed = vectorizer.fit_transform(features_train)
        features_test_transformed = vectorizer.transform(features_test)

        # feature selection, because text is super high dimensional and
        # can be really computationally chewy as a result
        selector = SelectPercentile(f_classif, percentile=1)
        selector.fit(features_train_transformed, labels_train)

        features_train_transformed = selector.transform(features_train_transformed).toarray()
        features_test_transformed = selector.transform(features_test_transformed).toarray()

        # info on the data
        print "no. of Chris training emails:", sum(labels_train)
        print "no. of Sara training emails:", len(labels_train) - sum(labels_train)

        self.dataset = {
            "features_train_transformed": features_train_transformed,
            "features_test_transformed": features_test_transformed,
            "labels_train": labels_train,
            "labels_test": labels_test
        }

