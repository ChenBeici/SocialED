SocialED
========

A Python Library for Social Event Detection
------------------------------------------

The field of Social Event Detection represents a pivotal area of research within the broader domains of artificial 
intelligence and natural language processing. Its objective is the automated identification and analysis of events from 
social media platforms such as Twitter and Facebook. Such events encompass a wide range of occurrences, including natural 
disasters and viral phenomena. 

To address this gap, we present Social Event Detection Python library called **SocialED**, an
**open-source Python library** designed to facilitate the development and evaluation of social event detection algorithms.

**SocialED stands out for**:

* **Broad spectrum of over 10 social event detection algorithms**, including classic techniques like Latent Dirichlet Allocation (LDA) and modern deep learning models such as BiLSTM, Word2Vec, GloVe, and more.
* **Unified APIs, comprehensive documentation, and practical examples** that enable users to format their data consistently, ensuring smooth integration with all social event detectors within SocialED.
* **Customizable and modular components** that empower users to tailor detection algorithms to meet specific requirements, facilitating the setup of social event detection workflows.
* **Rich utility functions** that streamline the process of building and executing social event detection tasks.
* **Reliable implementation** featuring unit tests, cross-platform continuous integration, as well as code coverage and maintainability assessments.

SocialED includes **10+** graph outlier detection algorithms.  
For consistency and accessibility, SocialED is developed on top of `DGL <https://www.dgl.ai/>`_ 
and `PyTorch <https://pytorch.org/>`_, and follows the API design of `PyOD <https://github.com/yzhao062/pyod>`_ 
and `PyGOD <https://github.com/pygod-team/pygod>`_.
See examples below for detecting outliers with SocialED in 5 lines!

Folder Structure
================

::

   .
   ├── LICENSE
   ├── MANIFEST.in
   ├── README.rst
   ├── docs
   ├── SocialED
   │   ├── __init__.py
   │   ├── datasets    
   │   ├── detector  
   │   └── metrics  
   ├── requirements.txt
   ├── setup.cfg
   └── setup.py

Installation
------------

It is recommended to use **pip** for installation.  
Please make sure **the latest version** is installed, as PyGOD is updated frequently:

.. code-block:: bash

   pip install SocialED           # normal install
   pip install --upgrade SocialED  # or update if needed

Alternatively, you could clone and run setup.py file:

.. code-block:: bash

    # Set up the environment
    conda create -n SocialED python=3.8
    conda activate SocialED

    # Installation
    git clone https://github.com/RingBDStack/SocialED.git
    cd SocialED
    pip install -r requirements.txt
    pip install .

**Required Dependencies**:

* python>=3.8
* numpy>=1.24.3
* scikit-learn>=1.2.2
* scipy>=1.10.1
* networkx>=2.3
* torch>=2.3.0
* torch_geometric>=2.5.3
* dgl>=0.6.0

API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Full API Reference: (https://socialed.readthedocs.io). API cheatsheet for all detectors:

* **preprocess()**: Preprocess the dataset.
* **fit()**: Fit the detector with train data.
* **detector()**: Initialize and configure the detection model, preparing it for training and prediction tasks.
* **evaluate(predictions, groundtruth)**: Assess the performance of the detector by comparing predictions with the actual data.

Usage & Example
---------------

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
   model.detection()

Collected Algorithms
--------------------

10+ different methods in total are implemented in this library. We provide an overview of their characteristics as follows.

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
- **CLKD**: Cross-lingual Knowledge Distillation (CLKD) combines a convolutional neural network with dynamic time warping to align sequences and detect events in streaming data. This online algorithm is effective for real-time applications.
- **MVGAN**: Multi-View Graph Attention Network (MVGAN) leverages multiple data views to enhance event detection accuracy. This offline algorithm uses GANs to model complex data distributions, improving robustness against noise and incomplete data.
- **KPGNN**: Knowledge-Preserving Graph Neural Network (KPGNN) is designed for incremental social event detection. It utilizes rich semantics and structural information in social messages to continuously detect events and extend its knowledge base. KPGNN outperforms baseline models, with potential for future research in event analysis and causal discovery in social data.
- **Finevent**: Fine-Grained Event Detection (FinEvent) uses a reinforced, incremental, and cross-lingual architecture for social event detection. It employs multi-agent reinforcement learning and density-based clustering (DRL-DBSCAN) to improve performance in various detection tasks. Future work will extend RL-guided GNNs for event correlation and evolution.
- **QSGNN**: Quality-Aware Self-Improving Graph Neural Network (QSGNN) improves open set social event detection with a pairwise loss and orthogonal constraint for training. It uses similarity distributions for pseudo labels and a quality-aware strategy to reduce noise, achieving state-of-the-art results in both closed and open set scenarios.
- **ETGNN**: Evidential Temporal-aware Graph Neural Network (ETGNN) enhances social event detection by integrating uncertainty and temporal information using Evidential Deep Learning and Dempster-Shafer theory. It employs a novel temporal-aware GNN aggregator, outperforming other methods.
- **HCRC**: Hybrid Graph Contrastive Learning for Social Event Detection (HCRC) captures comprehensive semantic and structural information from social messages. Using hybrid graph contrastive learning and reinforced incremental clustering, HCRC outperforms baselines across various experimental settings.
- **UCLSED**: Uncertainty-Guided Class Imbalance Learning Framework (UCLSED) enhances model generalization in imbalanced social event detection tasks. It uses an uncertainty-guided contrastive learning loss to handle uncertain classes and combines multi-view architectures with Dempster-Shafer theory for robust uncertainty estimation, improving detection accuracy.

Contact
-------

For further inquiries, please contact us via email at: [support@socialed.com](mailto:support@socialed.com)

