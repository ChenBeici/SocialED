# Configuration file for the Sphinx documentation builder.

import os
import sys
from os.path import dirname, abspath
from sphinx_gallery.sorting import FileNameSortKey

# 添加项目根目录到 Python 路径
root_dir = dirname(dirname(abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'SocialED'))

# Mock imports for modules that are difficult to install
autodoc_mock_imports = [
    # 基础依赖
    'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn',
    'networkx', 'tqdm',
    
    # PyTorch 相关
    'torch', 'torch.nn', 'torch.optim', 'torch.utils', 'torch.utils.data',
    'torchvision', 'pytorch_ignite',
    
    # 深度学习框架
    'dgl', 'dgl.function', 'dgl.dataloading',
    'torch_geometric',
    
    # NLP 相关
    'transformers', 'sentence_transformers', 'tokenizers',
    'spacy', 'gensim', 'en_core_web_lg', 'fr_core_news_lg',
    
    # 其他依赖
    'faiss', 'faiss_cpu',
    'git', 'GitPython',
    
    # SocialED 内部模块
    'SocialED.detector.bert',
    'SocialED.detector.sbert',
    'SocialED.detector.eventx',
    'SocialED.detector.clkd',
    'SocialED.detector.kpgnn',
    'SocialED.detector.finevent',
    'SocialED.detector.qsgnn',
    'SocialED.detector.hcrc',
    'SocialED.detector.uclsed',
    'SocialED.detector.rplmsed',
    'SocialED.detector.hisevent',
    'SocialED.detector.adpsemevent'
]

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

# Basic configuration
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'
root_doc = 'index'

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

# LaTeX output options
latex_documents = [
    (root_doc, 'socialED.tex', 'SocialED Documentation',
     'beici', 'manual'),
]

# Manual page output options
man_pages = [
    (root_doc, 'socialED', 'SocialED Documentation',
     [author], 1)
]

# Texinfo output options
texinfo_documents = [
    (root_doc, 'socialED', 'SocialED Documentation',
     author, 'SocialED', 'A Python library for social event detection.',
     'Miscellaneous'),
]
