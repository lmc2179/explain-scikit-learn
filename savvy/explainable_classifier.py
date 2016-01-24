import numpy as np
import copy
import random
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", context="talk")

class ExplainableClassifier(object):
    def __init__(self, feature_names, model):
        """
        Produce a wrapper around the model which allows explanations of particular classifications.

        Model should define the following, similar to scikit-learn classes:
            1. fit(X,y) method
            2. predict_proba(X)
            3. classes_ attribute
        """
        self.model = model
        self.feature_names = feature_names
        self.samplers = None

    def fit(self, X, y):
        [self._add_observations_to_sample(x) for x in X]
        self.model.fit(X, y)

    def partial_fit(self, X, y):
        [self._add_observations_to_sample(x) for x in X]
        self.model.partial_fit(X, y)

    def _add_observations_to_sample(self, x):
        if self.samplers:
            [sampler.observe(x_i) for sampler, x_i in zip(self.samplers, x)]
        else: # Perform initial construction of samplers by attempting type inference
            samplers = []
            for x_i in x:
                if isinstance(x_i, (int, float)) and not isinstance(x_i, bool):
                    sampler = _UniformRealSampler()
                else:
                    sampler = _UniformCategoricalSampler()
                sampler.observe(x_i)
                samplers.append(sampler)
            self.samplers = samplers

    def __getattr__(self, item):
        try:
            return getattr(self.model, item)
        except AttributeError:
            raise AttributeError

    def _sample_feature_contribution(self, feature, feature_permutations, input_sample, x):
        substitute_features = feature_permutations[:feature_permutations.index(feature)]
        perturbed_input_with_feature = self._substitute(x, input_sample, substitute_features + [feature])
        perturbed_input_without_feature = self._substitute(x, input_sample, substitute_features)
        predicted_probability_with_feature = self.model.predict_proba(perturbed_input_with_feature)
        predicted_probability_without_feature = self.model.predict_proba(perturbed_input_without_feature)
        feature_contribution_sample = predicted_probability_with_feature - predicted_probability_without_feature
        return feature_contribution_sample

    def _get_mean_of_samples(self, feature_contributions, number_of_samples):
        normalized_feature_contributions = {}
        for feature, contributions in feature_contributions.items():
            contribution_vector = contributions * (1.0 / number_of_samples)
            normalized_feature_contributions[feature] = {cls: contribution for cls, contribution in zip(self.classes_,
                                                                                             contribution_vector[0])}
        return normalized_feature_contributions

    def explain_classification(self, x, number_of_samples=1000):
        """Produce an explanation for the model's decision about the data point x. Returns an Explanation object,
        which will indicate the importance of each feature for each possible class in the model's decision."""
        feature_contributions = self._get_sum_of_sample_contributions(number_of_samples, x)
        normalized_feature_contributions = self._get_mean_of_samples(feature_contributions, number_of_samples)
        return Explanation(normalized_feature_contributions, x, self.model.predict_proba(x))

    def _get_sum_of_sample_contributions(self, number_of_samples, x):
        feature_contributions = {f: np.zeros((1, len(self.classes_))) for f in self.feature_names}
        feature_permutations = copy.deepcopy(self.feature_names)
        for i in range(number_of_samples):
            random.shuffle(feature_permutations)
            input_sample = self._sample_input_space()
            for feature in self.feature_names:
                feature_contribution_sample = self._sample_feature_contribution(feature, feature_permutations,
                                                                                input_sample, x)
                feature_contributions[feature] += feature_contribution_sample
        return feature_contributions

    def _sample_input_space(self):
        return [sampler.sample() for sampler in self.samplers]

    def _substitute(self, x, y, substituted_features):
        return [x[i] if feature in substituted_features else y[i] for i, feature in enumerate(self.feature_names)]

class _UniformCategoricalSampler(object):
    def __init__(self):
        self.observed = set()

    def observe(self, inp):
        self.observed.add(inp)

    def sample(self):
        return random.choice(list(self.observed))

class _UniformRealSampler(object):
    def __init__(self):
        self.min = None
        self.max = None

    def observe(self, inp):
        if self.min is None:
            self.min = inp
            self.max = inp
        else:
            self.min = min(self.min, inp)
            self.max = max(self.max, inp)

    def sample(self):
        return random.uniform(self.min, self.max)

class Explanation(object):
    "A plain old data object containing the explanation results."
    def __init__(self, feature_contributions, inputs, probabilistic_prediction):
        self.feature_names = np.array(list(feature_contributions.keys()))
        self.class_names = list(feature_contributions[self.feature_names[0]].keys())
        self.feature_contributions = feature_contributions
        self.inputs = inputs
        self.probabilistic_prediction = probabilistic_prediction

    def get_contribution(self, feature, cls):
        return self.feature_contributions[feature][cls]

    def get_feature_contribution_vector(self, feature):
        """Returns vector where each row is a contribution of the feature toward a class. Classes are represented
        in the same order as the class_names attribute."""
        return np.array([self.feature_contributions[feature][cls] for cls in self.classes_])

    def get_class_contribution_vector(self, cls):
        """Returns vector where each row is a contribution of a feature toward the class. Features are represented in the
        same order as the feature_names attribute.
        """
        return np.array([self.feature_contributions[f][cls] for f in self.feature_names])

def BarPlot(explanation):
    "Produce a number of barplots, one for each class."
    #TODO: Hide all this unwrapping of arrays for axes
    #TODO: Determination of y-axis dynamically?
    feature_names = explanation.feature_names
    x_axis = np.array(['{0}={1}'.format(f, str(explanation.inputs[i])) for i, f in enumerate(feature_names)])
    class_names = explanation.class_names
    f, *ax = plt.subplots(len(class_names), 1, figsize=(8, 6), sharex=True)
    for axis, cls in zip(ax[0], class_names):
        contribution_vector = explanation.get_class_contribution_vector(cls)
        sns.barplot(x_axis, contribution_vector, ci=None, hline=0, ax=axis)
        axis.set_ylabel('{0}'.format(cls))
    sns.despine(bottom=True)
    plt.setp(f.axes, yticks=[-1.0, -0.75,-0.5, 0.0, 0.5, 0.75, 1.0])
    plt.tight_layout(h_pad=3)
    plt.show()
