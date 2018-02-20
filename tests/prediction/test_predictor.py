#!/usr/bin/env python
# coding=utf-8

try:
    import mock

except ImportError:
    import unittest.mock as mock

from marvin_ml_intro_engine.prediction import Predictor


class TestPredictor:
    def test_execute(self, mocked_params):
        ac = Predictor(params=mocked_params)
        ac.execute(input_message="fake message")
        assert ac.params == mocked_params