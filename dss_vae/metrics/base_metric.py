""" define the basic interface of METRIC"""


class BaseMetric(object):
    def __init__(self, name='Base Metric'):
        self.name = name

    def _evaluating(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def preprocess(cls, **kwargs):
        raise NotImplementedError

    # def evaluate(self, **kwargs):
    #     score = self._evaluating(**kwargs)
    #     print("{} Score : ".format(self.name), score)
