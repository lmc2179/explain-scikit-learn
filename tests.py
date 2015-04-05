import unittest
import analyze
import pprint
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pandas as pd

#TODO: Separate explanation mechanism tests from visualization tests
#TODO: Add assertions to explanation mechanism tests
#TODO: Add test for Titanic data set

class AnalyzerTest(unittest.TestCase):
    @unittest.skip('Temp')
    def test_approximate_contribution(self):
        feature_names = ['feature', 'useless_1', 'useless_2']
        training_inputs = [[0,10,10],[10,10,10]]
        training_outputs = [0,1]
        model = LogisticRegression()
        analyzer = analyze.ExplainableClassifier(training_inputs, training_outputs, feature_names, model)
        explanations = [analyzer.explain_classification(inp, 1000) for inp in training_inputs]
        pprint.pprint(list(zip(training_inputs,explanations)))
        analyze.BarPlot(explanations[0])

    def test_iris(self):
        data = datasets.load_iris()
        model = LogisticRegression()
        target = [data.target_names[t] for t in data.target]
        analyzer = analyze.ExplainableClassifier(data.data, target, data.feature_names, model)
        explanation = analyzer.explain_classification(data.data[0])
        analyze.BarPlot(explanation)