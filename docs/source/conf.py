import os
import sys
from os.path import dirname, abspath

sys.path.insert(0, abspath('..'))
root_dir = dirname(dirname(abspath(__file__)))

project = 'SocialED'
copyright = '2024 SocialED Team'
author = 'SocialED Team'

version = '0.1.0'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    "sphinx.ext.mathjax",
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
    'sphinx_autodoc_typehints',
]

bibtex_bibfiles = ['references.bib']

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = "furo"
html_favicon = '_static/socialed.ico'
html_static_path = ['_static']

htmlhelp_basename = 'socialeddoc'

latex_elements = {}

latex_documents = [
    (master_doc, 'socialed.tex', 'SocialED Documentation',
     'SocialED Team', 'manual'),
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

from sphinx_gallery.sorting import FileNameSortKey

html_static_path = []

sphinx_gallery_conf = {
    'examples_dirs': '../examples/',
    'gallery_dirs': 'tutorials/',
    'within_subsection_order': FileNameSortKey,
    'filename_pattern': '.py',
    'download_all_examples': False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    'torch': ("https://pytorch.org/docs/master", None),
    'torch_geometric': ("https://pytorch-geometric.readthedocs.io/en/latest", None),
}

0

