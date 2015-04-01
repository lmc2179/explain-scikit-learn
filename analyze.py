import numpy as np
import copy
import random

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

    @property
    def class_names(self):
        return self.model.classes_

    def explain_classification(self, x, number_of_samples=1000):
        feature_contributions = {f:np.zeros(len(self.class_names)) for f in self.feature_names}
        feature_permutations = copy.deepcopy(self.feature_names)
        for i in range(number_of_samples):
            # Produce random elements (feature_permutations and input sample), calculate perturbed probability, adjust
            feature_permutations = random.shuffle(feature_permutations)
            input_sample = self._sample_input_space()
            for feature in self.feature_names:
                # Get next iteration of feature contribution vector
                substitute_features = feature_permutations[:feature_permutations.index(feature)]
                #TODO: Input vectors need better names
                perturbed_input_with_feature = self._tau(x, input_sample, substitute_features+[feature])
                perturbed_input_without_feature = self._tau(x, input_sample, substitute_features)
                v1 = self.model.predict_proba(perturbed_input_with_feature)
                v2 = self.model.predict_proba(perturbed_input_without_feature)
                feature_permutations[feature] += (v1 - v2)
        # Normalize each contribution vector by number of samples
        for feature, contributions in feature_contributions.values():
            feature_contributions[feature] = contributions * (1.0 / number_of_samples)
        return feature_contributions

    def _sample_input_space(self):
        return 0.0 #TODO: Basic input sampling scheme - uniform

    def _tau(self, x, y, features): #TODO: Rename: Name was chosen to match strumbelj et. al's notation in first version
        pass