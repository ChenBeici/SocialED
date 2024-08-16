.. SocialED documentation master file, created by
   sphinx-quickstart on Thu Jul  6 16:45:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root toctree directive.


.. figure:: socialED.png
    :scale: 30%
    :alt: logo


----

Social event detection is a crucial task in the field of natural language processing(NLP) and data mining, 
aimed at identifying, categorizing, and understanding events from a vast amount of unstructured text data, 
such as news articles, social media posts, and blogs. The primary goal is to detect and analyze events 
that have social significance, such as natural disasters, political movements, public health crises, 
and cultural events.

To address this gap, we present Social Event Detection Python library called **SocialED**, an
**open-source Python library** designed to facilitate the development and evaluation of social
event detection algorithms. 

**SocialED stands out for**:

* **Broad spectrum** of over 10 social event detection algorithms, including classic techniques like Latent Dirichlet Allocation (LDA) and modern deep learning models such as BiLSTM, Word2Vec, GloVe, and more.
* **Unified APIs, comprehensive documentation, and practical examples** that enable users to format their data consistently, ensuring smooth integration with all social event detectors within SocialED.
* **Customizable and modular components** that empower users to tailor detection algorithms to meet specific requirements, facilitating the setup of social event detection workflows.
* **Rich utility functions** that streamline the process of building and executing social event detection tasks.
* **Reliable implementation** featuring unit tests, cross-platform continuous integration, as well as code coverage and maintainability assessments.



SocialED includes **10+** graph outlier detection algorithms.
For consistency and accessibility, SocialED is developed on top of `DGL <https://www.dgl.ai/>`_ 
and `PyTorch <https://pytorch.org/>`_, and follows the API design of `PyOD <https://github.com/yzhao062/pyod>`_ 
and `PyGOD <https://github.com/pygod-team/pygod>`_.
See examples below for detecting outliers with SocialED in 5 lines!




**Social Event Detection Using SocialED with 8 Lines of Code**\ :

.. code-block:: python

   from socialed import KPGNN, args_define
   from data_sets import Event2012_Dataset

   # Load the dataset using the Event2012_Dataset class
   dataset = Event2012_Dataset.load_data()
   args = args_define.args

   # Create an instance of the KPGNN class with the parsed arguments and loaded dataset
   kpgnn = KPGNN(args, dataset)

   # Run the KPGNN instance
   kpgnn.preprocess()
   model = kpgnn.fit()
   kpgnn.detection()

----

Implemented Algorithms
----------------------

==================  =====  ========  =======  =============  =================================
Algorithm           Year   Type1     Type2    Type3          Ref
==================  =====  ========  =======  =============  =================================
LDA                 2003   Others    Offline  Supervised     :class:`socialed.detector.LDA`
BiLSTM              2005   Others    Offline  Supervised     :class:`socialed.detector.BiLSTM`
Word2Vec            2013   Others    Offline  Supervised     :class:`socialed.detector.Word2Vec`
GloVe               2014   Others    Offline  Supervised     :class:`socialed.detector.GloVe`
WMD                 2015   Others    Offline  Supervised     :class:`socialed.detector.WMD`
BERT                2018   PLM       Offline  Supervised     :class:`socialed.detector.BERT`
SBERT               2019   PLM       Offline  Supervised     :class:`socialed.detector.SBERT`
EventX              2020   Others    Online   Supervised     :class:`socialed.detector.EventX`
CLKD                2021   GNN       Online   Supervised     :class:`socialed.detector.CLKD`
KPGNN               2021   GNN       Online   Supervised     :class:`socialed.detector.KPGNN`
FinEvent            2022   GNN       Online   Supervised     :class:`socialed.detector.FinEvent`
QSGNN               2022   GNN       Online   Supervised     :class:`socialed.detector.QSGNN`
ETGNN               2023   GNN       Offline  Supervised     :class:`socialed.detector.ETGNN`
HCRC                2023   GNN       Online   Unsupervised   :class:`socialed.detector.HCRC`
UCLSED              2023   GNN       Offline  Supervised     :class:`socialed.detector.UCLSED`
RPLMSED             2024   PLM       Online   Supervised     :class:`socialed.detector.RPLMSED`
HISEvent            2024   Others    Online   Unsupervised   :class:`socialed.detector.HISEvent`
==================  =====  ========  =======  =============  =================================



----


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   tutorials/index
   api


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API References

   socialed.detector
   socialed.generator
   socialed.metric
   socialed.utils


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   cite
   team
   reference
