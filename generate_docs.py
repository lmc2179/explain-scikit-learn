from mimeograph import Generator
from savvy import explainable_classifier, regression_diagnostic

g = Generator()
g.generate_docs(project_name='savvy',
                summary="A collection of tools for interpreting and validating machine learning models.",
                modules=[explainable_classifier, regression_diagnostic],
                module_messages=['Turn any scikit-learn classifier into an interpretable model by adding a lightweight wrapper.',
                                 'Check the validity of your regression model.'])