# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from os.path import dirname, abspath

sys.path.insert(0, abspath('..'))
#sys.path.insert(0, abspath('../socialed'))
root_dir = dirname(dirname(abspath(__file__)))

# Verify if the path to your package is correct
print("Root dir:", root_dir)
print("System Path:", sys.path)

# -- Project information -----------------------------------------------------
# Debug import statements

# Debug import statements

project = 'socialed'
copyright = '2024, beici'
author = 'beici'
release = '0.1.0'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    #'sphinx.ext.autosummary',
    'sphinx.ext.doctest',   
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    "sphinx.ext.mathjax",
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'sphinx.ext.napoleon',
    #'sphinx_gallery.gen_gallery'
]

#autosummary_generate = True

autodoc_mock_imports = ['en_core_web_lg','argsparse']

bibtex_bibfiles = ['zreferences.bib']

templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'


exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output ----------------？---------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

pygments_style = 'sphinx'

html_theme = 'furo'
html_static_path = ['_static']


htmlhelp_basename = 'socialeddoc'

html_favicon = 'socialed.ico'


latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'socialed.tex', 'SocialED Documentation',
     'beici', 'manual'),
]

man_pages = [
    (master_doc, 'socialed', 'SocialED Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'socialed', 'SocialED Documentation',
     author, 'SocialED', 'One line description of project.',
     'Miscellaneous'),
]


html_static_path = []



intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info),
               None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    'torch': ("https://pytorch.org/docs/master", None),
    'torch_geometric': ("https://pytorch-geometric.readthedocs.io/en/latest",
                        None),
}




