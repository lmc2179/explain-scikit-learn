from setuptools import setup, find_packages

setup(
  name = 'savvy',
  packages = find_packages(),
  py_modules = ['savvy.analyze'],
  include_package_data=True,
  version = '0.0.2',
  description = 'Turn any scikit-learn classifier into an interpretable model by using a lightweight wrapper.',
  author = 'Louis Cialdella',
  author_email = 'louiscialdella@gmail.com',
  url = 'https://github.com/lmc2179/savvy',
  keywords = ['interpretable','transparent','machine learning','wrapper'],
  classifiers = [],
)
