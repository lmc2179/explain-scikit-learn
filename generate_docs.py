import pdoc
import os

DOCS_ROOT = 'docs'

TARGET_MODULES = ['regression_diagnostic', 'explainable_classifier']

def generate_doc(module_name):
    rd_doc = pdoc.html('savvy.{0}'.format(module_name))
    rd_doc_filename = os.path.join(DOCS_ROOT, module_name) + '.html'
    f = open(rd_doc_filename, 'w')
    f.write(rd_doc)
    f.close()

if __name__ == '__main__':
    for m in TARGET_MODULES:
        generate_doc(m)
