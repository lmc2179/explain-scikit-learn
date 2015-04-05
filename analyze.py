import numpy as np
import copy
import random
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", context="talk")

class ExplainableClassifier(object):
    def __init__(self, training_inputs, training_outputs, feature_names, model):
        """
        Train a model and store it so explanations can be produced later.

        Model should define the following, similar to scikit-learn classes:
            1. fit(X,y) method
            2. predict_proba(X)
            3. classes_ attribute
        """
        self.model = model
        self.feature_names = feature_names
        self.model.fit(training_inputs, training_outputs)
        self.training_min, self.training_max = self._get_vector_min_max(training_inputs)

    def _get_vector_min_max(self, vectors):
        min_vector = vectors[0]
        max_vector = vectors[0]
        for v in vectors[0:]:
            min_vector = [min(v1,v2) for v1, v2 in zip(v, min_vector)]
            max_vector = [max(v1,v2) for v1, v2 in zip(v, max_vector)]
        return min_vector, max_vector


    @property
    def class_names(self):
        return self.model.classes_

    def explain_classification(self, x, number_of_samples=1000):
        feature_contributions = {f:np.zeros((1,len(self.class_names))) for f in self.feature_names}
        feature_permutations = copy.deepcopy(self.feature_names)
        for i in range(number_of_samples):
            # Produce random elements (feature_permutations and input sample), calculate perturbed probability, adjust
            random.shuffle(feature_permutations)
            input_sample = self._sample_input_space()
            for feature in self.feature_names:
                # Get next iteration of feature contribution vector
                substitute_features = feature_permutations[:feature_permutations.index(feature)]
                #TODO: Input vectors need better names
                perturbed_input_with_feature = self._tau(x, input_sample, substitute_features+[feature])
                perturbed_input_without_feature = self._tau(x, input_sample, substitute_features)
                v1 = self.model.predict_proba(perturbed_input_with_feature)
                v2 = self.model.predict_proba(perturbed_input_without_feature)
                feature_contributions[feature] += (v1 - v2)
        # Normalize each contribution vector by number of samples
        for feature, contributions in feature_contributions.items():
            contribution_vector = contributions * (1.0 / number_of_samples)
            feature_contributions[feature] = {cls:contribution for cls, contribution in zip(self.class_names,
                                                                                            contribution_vector[0])}
        return feature_contributions

    def _sample_input_space(self):
        return [random.uniform(a,b) for a,b in zip(self.training_min, self.training_max)]

    def _tau(self, x, y, substituted_features): #TODO: Rename: Name was chosen to match strumbelj et. al's notation in first version
        return [x[i] if feature in substituted_features else y[i] for i, feature in enumerate(self.feature_names)]

def BarPlot(explanation):
    "Produce a number of barplots, one for each class."
    #TODO: Hide all this unwrapping of arrays
    #TODO: Cut this up into smaller functions, possibly embed in a class
    #TODO: Why is the X-axis not rendering correctly?
    #TODO: Determination of y-axis dynamically?
    feature_names = np.array(list(explanation.keys()))
    class_names = list(explanation[feature_names[0]].keys())
    f, *ax = plt.subplots(len(class_names), 1, figsize=(8, 6), sharex=True)
    for axis, cls in zip(ax[0], class_names):
        contribution_vector = np.array([explanation[f][cls] for f in feature_names])
        sns.barplot(feature_names, contribution_vector, ci=None, hline=0, ax=axis)
        axis.set_ylabel('{0}'.format(cls))
    sns.despine(bottom=True)
    plt.setp(f.axes, yticks=[-0.5, 0.0, 0.5])
    plt.tight_layout(h_pad=3)
    plt.show()
