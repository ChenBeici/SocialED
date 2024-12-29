# Configuration file for the Sphinx documentation builder.

import os
import sys
from os.path import dirname, abspath
from sphinx_gallery.sorting import FileNameSortKey

# 添加项目根目录和包目录到 Python 路径
root_dir = dirname(dirname(abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'SocialED'))

# Project information
project = 'SocialED'
copyright = '2024 beici'
author = 'beici'

# General configuration
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

# 添加 autosummary 配置
autosummary_generate = True
autosummary_imported_members = True
templates_path = ['_templates']

# Basic configuration
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'
master_doc = 'index'

# Mock imports
autodoc_mock_imports = [
    'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn',
    'networkx', 'tqdm', 'torch', 'transformers', 'dgl', 'spacy',
    'gensim', 'faiss', 'git'
]

# Bibliography
bibtex_bibfiles = ['zreferences.bib']

# HTML output options
html_theme = "furo"
html_favicon = 'socialed.ico'
html_static_path = ['_static']

# Sphinx Gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': 'examples/',
    'gallery_dirs': 'tutorials/',
    'within_subsection_order': FileNameSortKey,
    'filename_pattern': '.py',
    'download_all_examples': False,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    'torch': ("https://pytorch.org/docs/master", None),
    'torch_geometric': ("https://pytorch-geometric.readthedocs.io/en/latest", None),
}
