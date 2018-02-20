#!/usr/bin/env python
# coding=utf-8

try:
    import mock

except ImportError:
    import unittest.mock as mock

from marvin_ml_intro_engine.data_handler import AcquisitorAndCleaner


class TestAcquisitorAndCleaner:
    def test_execute(self, mocked_params):
        ac = AcquisitorAndCleaner(params=mocked_params)
        ac.execute()
        assert ac.params == mocked_params