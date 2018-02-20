#!/usr/bin/env python
# coding=utf-8

"""AcquisitorAndCleaner engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['AcquisitorAndCleaner']


logger = get_logger('acquisitor_and_cleaner')


class AcquisitorAndCleaner(EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(AcquisitorAndCleaner, self).__init__(**kwargs)

    def execute(self, **kwargs):
        from marvin_python_toolbox.common.data import MarvinData
        import pickle
        import cPickle

        # the words (features) and authors (labels), already largely preprocessed
        # this preprocessing will be repeated in the text learning mini-project

        print "Downloading files ...."
        authors_file_path = MarvinData.download_file(self.params["authors_file_path"])
        word_file_path = MarvinData.download_file(self.params["word_file_path"])

        print "Loading files ...."
        authors_file_handler = open(authors_file_path, "r")
        authors = pickle.load(authors_file_handler)
        authors_file_handler.close()

        words_file_handler = open(word_file_path, "r")
        word_data = cPickle.load(words_file_handler)
        words_file_handler.close()

        self.initial_dataset = {
            "word_data": word_data,
            "authors": authors
        }

        print "Done!"

