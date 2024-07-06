.. image:: https://github.com/ChenBeici/SocialED/blob/main/source/socialED.png?raw=true
   :target: https://github.com/ChenBeici/SocialED/blob/main/source/socialED.png?raw=true
   :width: 1050
   :alt: socialED Logo
   :align: center

SocialED
========

A Python Library for Social Event Detection

Social event detection is a critical task in natural language processing and has widespread
applications in various domains such as disaster response, public health monitoring, and
social media analysis. SocialED is an open-source Python library designed for social event
detection tasks in natural language processing. As the first comprehensive library of its
kind, SocialED supports a wide array of state-of-the-art methods for social event detection
under an easy-to-use, well-documented API tailored for both researchers and practitioners.
SocialED offers modularized components of various detection algorithms, allowing users to
easily customize each algorithm for their specific needs. To streamline the development
of detection workflows, SocialED provides numerous commonly used utility functions. To
handle large datasets efficiently. SocialED adheres to best practices in code reliability
and maintainability, including unit testing, continuous integration, and code coverage. To
ensure accessibility, SocialED is released under a permissive BSD 2-Clause license and
is available at https://github.com/RingBDStack/socialED and on the Python Package
Index (PyPI).

+-------------------------------------+
|               SocialED              |
+-------------------------------------+

+---------------------------------+  +---------------------------------+  +---------------------------------+
|       Traditional Algorithms    |  |        GNN-based Algorithms     |  |        PLM-based Algorithms     |
|  - LDA                          |  |  - CLKD                         |  |  - RPLMsed                      |
|  - BiLSTM                       |  |  - MVGAN                        |  |                                 |
|  ...                            |  |  ...                            |  |                                 |
|  - EventX                       |  |  - KPGNN                        |  |                                 |
+---------------------------------+  +---------------------------------+  +---------------------------------+
            |                                         |                                      |
            v                                         v                                      v
+-----------------------+                +-----------------------+               +-----------------------+
|         prediction()  |                |         fit()         |               |         fit()         |
+-----------------------+                +-----------------------+               +-----------------------+
                                         +-----------------------+               +-----------------------+
                                         |     prediction()      |               |     prediction()      |
                                         +-----------------------+               +-----------------------+

+-----------------------+                +-----------------------+               +-----------------------+
|         prediction()  |                |         fit()         |               |         fit()         |
+-----------------------+                +-----------------------+               +-----------------------+
                                         +-----------------------+               +-----------------------+
                                         |     prediction()      |               |     prediction()      |
                                         +-----------------------+               +-----------------------+

+-------------------+                       +-------------------+                   +-------------------+
|     Datasets      |                       | Evaluation Metrics|                   |     Model Tasks   |
|  - Events2012     |                       |  - NMI            |                   |  - Training       |
|  - Events2018     |                       |  - ARI            |                   |  - Prediction     |
|  ...              |                       |  ...              |                   +-------------------+
|  - MAVEN          |                       |  - F1-score       |
+-------------------+                       +-------------------+

Folder Structure
----------------

::

    .
    ├── build
    ├── dist
    ├── docs
    ├── examples
    │   └── KPGNN_example.py
    ├── socialED
    │   ├── datasets
    │   │   ├── __init__.py
    │   │   ├── data
    │   │   ├── ACE2005.py
    │   │   ├── Arabic_Twitter.py
    │   │   ├── CrisisLexT26.py
    │   │   ├── CrisisLexT6.py
    │   │   ├── Event2012.py
    │   │   ├── Event2018.py
    │   │   ├── MAVEN.py
    │   │   └── __pycache__
    │   ├── detector
    │   │   ├── __init__.py
    │   │   ├── 1-LDA
    │   │   ├── 2-BiLSTM
    │   │   ├── 3-word2vec
    │   │   ├── 4-glove
    │   │   ├── 5-WMD
    │   │   ├── 6-bert
    │   │   ├── 7-sbert
    │   │   ├── 8-EventX
    │   │   ├── 9-CLKD
    │   │   ├── 11-PPGCN
    │   │   ├── 11-PPGCNS
    │   │   ├── 12-KPGNN
    │   │   ├── 13-FinEvent
    │   │   ├── 14-QSGNN
    │   │   ├── 17-UCL_SED
    │   │   ├── 18-RPLM_SED
    │   │   ├── 19-HISEvent
    │   │   └── __pycache__
    │   ├── __init__.py
    │   ├── metrics
    │   ├── __pycache__
    │   └── utils
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── setup.cfg
    ├── setup.py
    └── socialED.egg-info

Installation
------------

### Manually

.. code-block:: bash

    # Set up the environment
    conda create -n socialED python=3.8
    conda activate socailED

    # Installation
    git clone https://github.com/yukobebryantlakers/socialED.git
    pip install -r requirements.txt
    pip install socialED

Usage & Example
---------------

.. code-block:: python

    from socialED import KPGNN, args_define
    from Event2012 import Event2012_Dataset

    # Load the dataset using the Event2012_Dataset class
    dataset = Event2012_Dataset.load_data()
    args = args_define.args

    # Create an instance of the KPGNN class with the parsed arguments and loaded dataset
    kpgnn = KPGNN(args, dataset)

    # Run the KPGNN instance
    kpgnn.run()

Collected Algorithms
--------------------

19 different methods in total are implemented in this library. We provide an overview of their characteristics as follows.

Algorithm Descriptions
----------------------

- **LDA**: Latent Dirichlet Allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups. It is particularly useful for discovering the hidden thematic structure in large text corpora.
- **BiLSTM**: Bi-directional Long Short-Term Memory (BiLSTM) networks enhance the capabilities of traditional LSTMs by processing sequences in both forward and backward directions. This bidirectional approach is effective for tasks like sequence classification and time series prediction.
- **Word2Vec**: Word2Vec is a family of models that generate word embeddings by training shallow neural networks to predict the context of words. These embeddings capture semantic relationships between words, making them useful for various natural language processing tasks.
- **GLOVE**: Global Vectors for Word Representation (GLOVE) generates word embeddings by aggregating global word-word co-occurrence statistics from a corpus. This approach produces vectors that capture meaning effectively, based on the frequency of word pairs in the training text.
- **WMD**: Word Mover's Distance (WMD) measures the semantic distance between two documents by computing the minimum distance that words from one document need to travel to match words from another document. This method is grounded in the concept of word embeddings.
- **BERT**: Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based model that pre-trains deep bidirectional representations by conditioning on both left and right context in all layers. BERT has achieved state-of-the-art results in many NLP tasks.
- **SBERT**: Sentence-BERT (SBERT) modifies the BERT network to generate semantically meaningful sentence embeddings that can be compared using cosine similarity. It is particularly useful for sentence clustering and semantic search.
- **EventX**: EventX is designed for online event detection in social media streams, processing tweets in real-time to identify emerging events by clustering similar content. This framework is optimized for high-speed data environments.
- **CLKD**: cross-lingual knowledge distillation (CLKD) combines a convolutional neural network with dynamic time warping to align sequences and detect events in streaming data. This online algorithm is effective for real-time applications.
- **MVGAN**: Multi-View Graph Attention Network (MVGAN) leverages multiple data views to enhance event detection accuracy. This offline algorithm uses GANs to model complex data distributions, improving robustness against noise and incomplete data.

We provide their statistics as follows.

+-------------+--------+---------+-------------+---------------------------+
| Algorithm   | Type1  | Type2   | Type3       | Reference                 |
+=============+========+=========+=============+===========================+
| **LDA**     | Others | Offline | Supervised  | (David M. Blei et al. 2003)|
+-------------+--------+---------+-------------+---------------------------+
| **BiLSTM**  | Others | Offline | Supervised  | (Alex Graves et al. 2005)  |
+-------------+--------+---------+-------------+---------------------------+
| **Word2Vec**| Others | Offline | Supervised  | (Tomas Mikolov et al. 2013)|
+-------------+--------+---------+-------------+---------------------------+
| **GLOVE**   | Others | Offline | Supervised  | (Jeffrey Pennington et al. 2014) |
+-------------+--------+---------+-------------+---------------------------+
| **WMD**     | Others | Offline | Supervised  | (Matt Kusner et al. 2015)  |
+-------------+--------+---------+-------------+---------------------------+
| **BERT**    | PLM    | Offline | Supervised  | (J. Devlin et al. 2018)    |
+-------------+--------+---------+-------------+---------------------------+
| **SBERT**   | PLM    | Offline | Supervised  | (Nils Reimers et al. 2019) |
+-------------+--------+---------+-------------+---------------------------+
| **EventX**  | Others | Online  | Supervised  | (BANG LIU et al. 2020)     |
+-------------+--------+---------+-------------+---------------------------+
| **CLKD**    | GNN    | Online  | Supervised  | (Jiaqian Ren et al. 2021)  |
+-------------+--------+---------+-------------+---------------------------+
| **MVGAN**   | GNN    | Offline | Supervised  | (Wanqiu Cui et al. 2021)   |
+-------------+--------+---------+-------------+---------------------------+
| **PP-GCN**  | GNN    | Online  | Supervised  | (Hao Peng et al. 2021)     |
+-------------+--------+---------+-------------+---------------------------+
| **KPGNN**   | GNN    | Online  | Supervised  | (Yuwei Cao et al. 2021)    |
+-------------+--------+---------+-------------+---------------------------+
| **Finevent**| GNN    | Online  | Supervised  | (Hao Peng et al. 2022)     |
+-------------+--------+---------+-------------+---------------------------+
| **QSGNN**   | GNN    | Online  | Supervised  | (Jiaqian Ren et al. 2022)  |
+-------------+--------+---------+-------------+---------------------------+
| **ETGNN**   | GNN    | Offline | Supervised  | (Jiaqian Ren et al. 2023)  |
+-------------+--------+---------+-------------+---------------------------+
| **HCRC**    | GNN    | Online  | Unsupervised| (Yuanyuan Guo et al. 2023) |
+-------------+--------+---------+-------------+---------------------------+
| **UCLsed**  | GNN    | Offline | Supervised  | (Jiaqian Ren et al. 2023)  |
+-------------+--------+---------+-------------+---------------------------+
| **RPLMsed** | PLM    | Online  | Supervised  | (Pu Li et al. 2024)        |
+-------------+--------+---------+-------------+---------------------------+
| **HISevent**| Others | Online  | Unsupervised| (Yuwei Cao et al. 2024)    |
+-------------+--------+---------+-------------+---------------------------+

Collected Datasets
------------------

-   **ACE2005**: The ACE2005 dataset is a comprehensive collection of news articles annotated for entities, relations, and events. It includes a diverse range of event types and is widely used for event extraction research.
-   **MAVEN**: MAVEN (MAssive eVENt) is a large-scale dataset for event detection that consists of over 11,000 events annotated across a wide variety of domains. It is designed to facilitate the development of robust event detection models.
-   **TAC KBP**: The TAC KBP dataset is part of the Text Analysis Conference Knowledge Base Population track. It contains annotated events, entities, and relations, focusing on extracting structured information from unstructured text.
-   **CrisisLexT26**: CrisisLexT26 is a dataset containing tweets related to 26 different crisis events. It is used to study information dissemination and event detection in social media during emergencies.
-   **CrisisLexT6**: CrisisLexT6 is a smaller dataset from the CrisisLex collection, focusing on six major crisis events. It includes annotated tweets that provide valuable insights into public response and information spread during crises.
-   **Event2012**: Event2012 is a dataset composed of tweets related to various events in 2012. It includes a wide range of event types and is used for studying event detection and classification in social media.
-   **Event2018**: Event2018 consists of French tweets annotated for different event types. It provides a multilingual perspective on event detection, allowing researchers to explore language-specific challenges and solutions.
-   **KBP2017**: KBP2017 is part of the Knowledge Base Population track and focuses on extracting entities, relations, and events from text. It is a valuable resource for developing and benchmarking information extraction systems.
-   **CySecED**: CySecED is a dataset designed for cybersecurity event detection. It includes annotated cybersecurity events and is used to study threat detection and response in textual data.
-   **FewED**: FewED is a dataset for few-shot event detection, providing a limited number of annotated examples for each event type. It is designed to test the ability of models to generalize from few examples.

We provide their statistics as follows.

+----------------+--------+--------------+-----------+-----------+-----------+
| Dataset        | Events | Event_Types  | Sentences | Tokens    | Documents |
+================+========+==============+===========+===========+===========+
| **ACE2005**    | 5,349  | 33           | 11,738    | 230,382   | 599       |
+----------------+--------+--------------+-----------+-----------+-----------+
| **MAVEN**      | 11,191 | 168          | 23,663    | 512,394   | 4,480     |
+----------------+--------+--------------+-----------+-----------+-----------+
| **TAC KBP**    | 3,500  | 18           | 7,800     | 150,000   | 2,500     |
+----------------+--------+--------------+-----------+-----------+-----------+
| **CrisisLexT26**| 4,353 | 26           | 8,000     | 175,000   | 1,200     |
+----------------+--------+--------------+-----------+-----------+-----------+
| **CrisisLexT6**| 2,100  | 6            | 4,500     | 90,000    | 600       |
+----------------+--------+--------------+-----------+-----------+-----------+
| **Event2012**  | 68,841 | 20           | 150,000   | 3,000,000 | 10,000    |
+----------------+--------+--------------+-----------+-----------+-----------+
| **Event2018**  | 15,000 | 10           | 50,000    | 1,000,000 | 5,000     |
+----------------+--------+--------------+-----------+-----------+-----------+
| **KBP2017**    | 4,200  | 22           | 9,000     | 180,000   | 3,000     |
+----------------+--------+--------------+-----------+-----------+-----------+
| **CySecED**    | 5,500  | 35           | 12,000    | 250,000   | 4,200     |
+----------------+--------+--------------+-----------+-----------+-----------+
| **FewED**      | 6,000  | 40           | 14,000    | 300,000   | 5,500     |
+----------------+--------+--------------+-----------+-----------+-----------+

How to Contribute
-----------------

You are welcome to become part of this project. See `contribute guide <./docs/contribute.md>`_ for more information.

Authors & Acknowledgements
--------------------------

Contact
-------

Reach out to us by submitting an issue report or sending an email to sy2339225@buaa.edu.

References
----------
