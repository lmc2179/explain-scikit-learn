import unittest
import pprint
from collections import defaultdict
from functools import reduce

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

from explain_sklearn import analyze


#TODO: Separate explanation mechanism tests from visualization tests
#TODO: Add assertions to explanation mechanism tests
#TODO: Add test for Titanic data set

class AnalyzerTest(unittest.TestCase):
    @unittest.skip('')
    def test_approximate_contribution(self):
        feature_names = ['feature', 'useless_1', 'useless_2']
        training_inputs = [[0,10,10],[10,10,10]]
        training_outputs = [0,1]
        model = LogisticRegression()
        explainable_model = analyze.ExplainableClassifier(feature_names, model)
        explainable_model.fit(training_inputs, training_outputs)
        explanations = [explainable_model.explain_classification(inp, 1000) for inp in training_inputs]
        pprint.pprint(list(zip(training_inputs,explanations)))
        analyze.BarPlot(explanations[0])

    @unittest.skip('')
    def test_iris(self):
        data = datasets.load_iris()
        model = LogisticRegression()
        target = [data.target_names[t] for t in data.target]
        explainable_model = analyze.ExplainableClassifier(data.feature_names, model)
        explainable_model.fit(data.data, target)
        explanation = explainable_model.explain_classification(data.data[0])
        analyze.BarPlot(explanation)

    @unittest.skip('')
    def test_categorical(self): #To become Titanic test; needs categorical data model
        model = CategoricalNaiveBayes()
        explainable_model = analyze.ExplainableClassifier(['useful', 'useless'], model)
        data = [['A','C'],['B', 'C']]
        target = ['A', 'B']
        explainable_model.fit(data, target)
        explanation = explainable_model.explain_classification(data[0])
        analyze.BarPlot(explanation)

    def test_boolean_or(self):
        model = BooleanOrTestModel()
        explainable_model = analyze.ExplainableClassifier(['1','2'], model)
        data = [[True, True],[True, False],[False, True],[False, False]]
        target = [True, True, True, False]
        explainable_model.fit(data, target)
        explanation = explainable_model.explain_classification(data[0], number_of_samples=10000)
        tolerance = 0.05
        actual_contribution = 1/8.0
        lower_bound = actual_contribution - tolerance
        upper_bound = actual_contribution + tolerance
        true_contributions = explanation.get_class_contribution_vector(True)
        print(true_contributions)
        assert all([lower_bound < v_i< upper_bound for v_i in true_contributions])

class BooleanOrTestModel(object):
    def fit(self, X,y):
        pass

    @property
    def classes_(self):
        return [True, False]

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        return np.array([[self.predict_proba_for_class(x, cls) for cls in self.classes_] for x in X])

    def predict_proba_for_class(self, x, cls):
        return 1.0 if any(x)==cls else 0.0

class CategoricalNaiveBayes(object):
    def __init__(self):
        self.class_totals = defaultdict(int)
        self.counts = defaultdict(int)

    @property
    def classes_(self):
        return list(self.class_totals.keys())

    def fit(self, X, y):
        for input_point, output_class in zip(X, y):
            self.class_totals[output_class] += 1
            for i, x_i in enumerate(input_point):
                self.counts[(i,x_i, output_class)] += 1

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        return np.array([[self.predict_proba_for_class(x, cls) for cls in self.classes_] for x in X])

    def predict_proba_for_class(self, x, cls):
        independent_probabilities = []
        for i, x_i in enumerate(x):
            independent_probabilities.append(self.counts[(i, x_i, cls)]/self.class_totals[cls])
        return reduce(lambda p1, p2: p1*p2, independent_probabilities)