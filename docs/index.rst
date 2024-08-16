.. SocialED documentation master file, created by
   sphinx-quickstart on [日期].
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. figure:: SocialED.png
    :scale: 30%
    :alt: logo

----


.. image:: https://img.shields.io/pypi/v/socialed.svg?color=brightgreen
   :target: https://pypi.org/project/socialed/
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/socialed/badge/?version=latest
   :target: https://docs.socialed.org/en/latest/?badge=latest
   :alt: Documentation status

.. image:: https://img.shields.io/github/stars/socialed-team/socialed.svg
   :target: https://github.com/ChenBeici/SocialED/stargazers
   :alt: GitHub stars

.. image:: https://img.shields.io/github/forks/socialed-team/socialed.svg?color=blue
   :target: https://github.com/ChenBeici/SocialED/network
   :alt: GitHub forks

.. image:: https://static.pepy.tech/personalized-badge/socialed?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
   :target: https://pypi.org/project/SocialED/
   :alt: PyPI downloads

.. image:: https://github.com/ChenBeici/SocialED/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/ChenBeici/SocialED/actions/workflows/testing.yml
   :alt: testing

.. image:: https://coveralls.io/repos/github/socialed-team/socialed/badge.svg?branch=main
   :target: https://coveralls.io/github/socialed-team/socialed?branch=main
   :alt: Coverage Status

.. image:: https://img.shields.io/github/license/socialed-team/socialed.svg
   :target: https://github.com/ChenBeici/SocialED/blob/master/LICENSE
   :alt: License

----

SocialED is a **Python library** for **social event detection**.
This field has critical applications, such as detecting events from social media streams and identifying patterns in large-scale social interactions.

SocialED includes **10+** social event detection algorithms.
For consistency and accessibility, SocialED is developed on top of `PyTorch <https://pytorch.org/>`_ and other popular libraries, ensuring ease of use and integration.


**SocialED is featured for**:

* **Broad spectrum** of over 10 social event detection algorithms, including classic techniques like Latent Dirichlet Allocation (LDA) and modern deep learning models such as BiLSTM, Word2Vec, GloVe, and more.
* **Unified APIs, comprehensive documentation, and practical examples** that enable users to format their data consistently, ensuring smooth integration with all social event detectors within SocialED.
* **Customizable and modular components** that empower users to tailor detection algorithms to meet specific requirements, facilitating the setup of social event detection workflows.
* **Rich utility functions** that streamline the process of building and executing social event detection tasks.
* **Reliable implementation** featuring unit tests, cross-platform continuous integration, as well as code coverage and maintainability assessments.


**Social Event Detection Using SocialED with 5 Lines of Code**\ :


.. code-block:: python

   from SocialED.detector import KPGNN
   from SocialED.data import Event2012_Dataset

   # Load the dataset using the Event2012_Dataset class
   dataset = Event2012_Dataset.load_data()

   # Create an instance of the KPGNN class and loaded dataset
   model = KPGNN(dataset)

   # Run the KPGNN instance
   model.preprocess()
   model = model.fit()
   predictions, groundtruth = model.detection()


----


Implemented Algorithms
----------------------

==================  =====  ==========  ==========  ============  =========================
Algorithm           Year   Category    Environment  Supervision   Ref
==================  =====  ==========  ==========  ============  =========================
LDA                 2003   Others      Offline      Supervised     [#Blei2003lda]_
BiLSTM              2005   Others      Offline      Supervised     [#Graves2005bilstm]_
Word2Vec            2013   Others      Offline      Supervised     [#Mikolov2013word2vec]_
GloVe               2014   Others      Offline      Supervised     [#Pennington2014glove]_
WMD                 2015   Others      Offline      Supervised     [#Kusner2015wmd]_
BERT                2018   PLM         Offline      Supervised     [#Devlin2018bert]_
SBERT               2019   PLM         Offline      Supervised     [#Reimers2019sbert]_
EventX              2020   Others      Online       Supervised     [#Liu2020eventx]_
CLKD                2021   GNN         Online       Supervised     [#Ren2021clkd]_
KPGNN               2021   GNN         Online       Supervised     [#Cao2021kpgnn]_
FinEvent            2022   GNN         Online       Supervised     [#Peng2022finevent]_
QSGNN               2022   GNN         Online       Supervised     [#Ren2022qsgnn]_
ETGNN               2023   GNN         Offline      Supervised     [#Ren2023etgnn]_
HCRC                2023   GNN         Online       Unsupervised   [#Guo2023hcrc]_
UCLSED              2023   GNN         Offline      Supervised     [#Ren2023uclsad]_
RPLMSED             2024   PLM         Online       Supervised     [#Li2024rplmsed]_
HISEvent            2024   Others      Online       Unsupervised   [#Cao2024hisevent]_
==================  =====  ==========  ==========  ============  ==========================



----


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   tutorials/index
   quickstart
   examples

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: API References

   socialed.detector
   socialed.dataset
   socialed.metrics
   socialed.utils
   socialed.tests

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   cite
   team
   reference
