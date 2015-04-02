import unittest
import analyze
import pprint
from sklearn.linear_model import LogisticRegression

class AnalyzerTest(unittest.TestCase):
    def test_approximate_contribution(self):
        feature_names = ['feature', 'useless_1', 'useless_2']
        training_inputs = [[0,10,10],[10,10,10]]
        training_outputs = [0,1]
        model = LogisticRegression()
        analyzer = analyze.ExplainableClassifier(training_inputs, training_outputs, feature_names, model)
        explanations = [analyzer.explain_classification(inp, 1000) for inp in training_inputs]
        pprint.pprint(list(zip(training_inputs,explanations)))
        analyze.BarPlot(explanations[0])