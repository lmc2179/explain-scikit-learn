import numpy as np
import copy
import random

class ClassificationExplanationModel(object):
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
        feature_contributions = {} #Each value is a vector of class contributions
        permutations = copy.deepcopy(self.feature_names)
        for i in range(number_of_samples):
            # Produce random elements (permutations and input sample), calculate perturbed probability, adjust
            permutations = random.shuffle(permutations)
            for feature in self.feature_names:
                # Get next iteration of feature contribution vector
                pass
        # Normalize each contribution vector by number of samples
        for feature, contributions in feature_contributions.values():
            feature_contributions[feature] = contributions * (1.0 / number_of_samples)
        return feature_contributions

    def _produce_input_sample(self):
        pass