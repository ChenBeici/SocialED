# Configuration file for the Sphinx documentation builder.

import os
import sys
from os.path import dirname, abspath

# 将 SocialED 项目的根目录添加到 sys.path
sys.path.insert(0, abspath('..'))
root_dir = dirname(dirname(abspath(__file__)))

# -- Project information -----------------------------------------------------

project = 'SocialED'
copyright = '2024 beici'
author = 'beici'

version_path = os.path.join(root_dir, 'SocialED', 'version.py')
exec(open(version_path).read())
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

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
    'sphinx_gallery.gen_gallery'
]

bibtex_bibfiles = ['zreferences.bib']

autodoc_mock_imports = [
    'en_core_web_lg',
    'fr_core_news_lg',
    'dgl',
    'dgl.function',
    'dgl.dataloading',
    'spacy',
    'torch',
    'torch.nn',
    'torch.cuda',
    'transformers',
    'transformers.modeling_bert',
    'transformers.tokenization_bert',
    'numpy',
    'pandas',
    'scikit-learn'
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_favicon = 'socialed.ico'
html_static_path = ['_static']

# -- Options for HTMLHelp output ---------------------------------------------
htmlhelp_basename = 'socialEDdoc'

# -- Options for LaTeX output ------------------------------------------------
latex_documents = [
    (master_doc, 'socialED.tex', 'SocialED Documentation',
     'beici', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, 'socialED', 'SocialED Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, 'socialED', 'SocialED Documentation',
     author, 'SocialED', 'A Python library for social event detection.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------
from sphinx_gallery.sorting import FileNameSortKey

html_static_path = []



sphinx_gallery_conf = {
    'examples_dirs': 'examples/',   # Path to your example scripts
    'gallery_dirs': 'tutorials/',
    'within_subsection_order': FileNameSortKey,
    'filename_pattern': '.py',
    'download_all_examples': False,
}

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    'torch': ("https://pytorch.org/docs/master", None),
    'torch_geometric': ("https://pytorch-geometric.readthedocs.io/en/latest", None),
}
