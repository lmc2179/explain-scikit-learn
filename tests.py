import unittest
import analyze
from sklearn.linear_model import LogisticRegression

class AnalyzerTest(unittest.TestCase):
    def test_approximate_contribution(self):
        feature_names = ['featuer', 'useless_1', 'useless_2']
        training_inputs = [[0,10,10],[10,10,10]]
        training_outputs = [0,1]
        model = LogisticRegression()
        analyzer = analyze.ClassificationExplanation(training_inputs, training_outputs, feature_names, model)
        for inp in training_inputs:
            print(analyzer.explain_classification(inp, 0))
            print(analyzer.explain_classification(inp, 0))
