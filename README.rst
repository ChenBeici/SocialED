.. image:: https://github.com/ChenBeici/SocialED/blob/main/source/SocialED.png?raw=true
   :target: https://github.com/ChenBeici/SocialED/blob/main/source/SocialED.png?raw=true
   :width: 1050
   :alt: SocialED Logo
   :align: center


.. image:: https://readthedocs.org/projects/pygod/badge/?version=latest
   :target: https://socialed.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status

.. image:: https://img.shields.io/github/stars/ChenBeici/SocialED.svg
   :target: https://github.com/ChenBeici/SocialED/stargazers
   :alt: GitHub stars

.. image:: https://img.shields.io/github/forks/ChenBeici/SocialED.svg?color=blue
   :target: https://github.com/ChenBeici/SocialED/network
   :alt: GitHub forks

.. image:: https://github.com/ChenBeici/SocialED/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/ChenBeici/SocialED/actions/workflows/testing.yml
   :alt: testing


.. image:: https://img.shields.io/github/license/ChenBeici/SocialED.svg
   :target: https://github.com/ChenBeici/SocialED/blob/master/LICENSE
   :alt: License


-----



SocialED
========

A Python Library for Social Event Detection

The field of Social Event Detection (SED) represents a pivotal area of research within the broader domains of artificial 
intelligence and natural language processing. Its objective is the automated identification and analysis of events from 
social media platforms such as Twitter and Facebook. Such events encompass a wide range of occurrences, including natural 
disasters and viral phenomena. 
The objective of SED is to detect, classify and comprehend these events in real-time by processing vast quantities of 
unstructured data through techniques such as machine learning, text mining and network analysis. This is crucial for 
applications such as crisis management, market analysis and public sentiment monitoring, where timely and accurate 
event detection facilitates informed decision-making.

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



Folder Structure
================

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

### Manually

.. code-block:: bash

    # Set up the environment
    conda create -n SocialED python=3.8
    conda activate SocialED

    # Installation
    git clone https://github.com/ChenBeici/SocialED.git
    pip install -r requirements.txt
    pip install SocialED

**Required Dependencies**\ :

* python>=3.8
* numpy>=1.24.3
* scikit-learn>=1.2.2
* scipy>=1.10.1
* networkx>=2.3
* torch>=2.0.0
* torch_geometric>=2.3.0


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
- **CLKD**: cross-lingual knowledge distillation (CLKD) combines a convolutional neural network with dynamic time warping to align sequences and detect events in streaming data. This online algorithm is effective for real-time applications.
- **MVGAN**: Multi-View Graph Attention Network (MVGAN) leverages multiple data views to enhance event detection accuracy. This offline algorithm uses GANs to model complex data distributions, improving robustness against noise and incomplete data.
- **KPGNN**: Knowledge-Preserving Graph Neural Network (KPGNN) is designed for incremental social event detection. It utilizes rich semantics and structural information in social messages to continuously detect events and extend its knowledge base. KPGNN outperforms baseline models, with potential for future research in event analysis and causal discovery in social data.
- **Finevent**: Fine-Grained Event Detection (FinEvent) uses a reinforced, incremental, and cross-lingual architecture for social event detection. It employs multi-agent reinforcement learning and density-based clustering (DRL-DBSCAN) to improve performance in various detection tasks. Future work will extend RL-guided GNNs for event correlation and evolution.
- **QSGNN**: Quality-Aware Self-Improving Graph Neural Network (QSGNN) improves open set social event detection with a pairwise loss and orthogonal constraint for training. It uses similarity distributions for pseudo labels and a quality-aware strategy to reduce noise, achieving state-of-the-art results in both closed and open set scenarios.
- **ETGNN**: Evidential Temporal-aware Graph Neural Network (ETGNN) enhances social event detection by integrating uncertainty and temporal information using Evidential Deep Learning and Dempster-Shafer theory. It employs a novel temporal-aware GNN aggregator, outperforming other methods.
- **HCRC**: Hybrid Graph Contrastive Learning for Social Event Detection (HCRC) captures comprehensive semantic and structural information from social messages. Using hybrid graph contrastive learning and reinforced incremental clustering, HCRC outperforms baselines across various experimental settings.
- **UCLSED**: Uncertainty-Guided Class Imbalance Learning Framework (UCLSED) enhances model generalization in imbalanced social event detection tasks. It uses an uncertainty-guided contrastive learning loss to handle uncertain classes and combines multi-view architectures with Dempster-Shafer theory for robust uncertainty estimation, achieving superior results.
- **RPLMSED**: Relational Prompt-Based Pre-Trained Language Models for Social Event Detection (RPLMSED) uses pairwise message modeling to address missing and noisy edges in social message graphs. It leverages content and structural information with a clustering constraint to enhance message representation, achieving state-of-the-art performance in various detection tasks.
- **HISevent**: Structural Entropy-Based Social Event Detection (HISevent) is an unsupervised tool that explores message correlations without the need for labeling or predetermining the number of events. HISevent combines GNN-based methods' advantages with efficient and robust performance, achieving new state-of-the-art results in closed- and open-set settings.



We provide their statistics as follows.

==================  =====  ==========  ==========  ============  =====================
Algorithm           Year   Category    Environment  Supervision   Ref
==================  =====  ==========  ==========  ============  =====================
LDA                 2003   Others      Offline      Supervised     [#Blei2003lda]_
BiLSTM              2005   Others      Offline      Supervised     [#Graves2005bilstm]_
Word2Vec            2013   Others      Offline      Supervised     [#Mikolov2013word2vec]_
GloVe               2014   Others      Offline      Supervised     [#Pennington2014glove]_
WMD                 2015   Others      Offline      Supervised     [#Kusner2015wmd]_
BERT                2018   PLM         Offline      Supervised     [#Devlin2018bert]_
SBERT               2019   PLM         Offline      Supervised     [#Reimers2019sbert]_
EventX              2020   Others      Online       Supervised     [#Liu2020eventx]_
CLKD                2021   GNN         Online       Supervised     [#Ren2021clkd]_
MVGAN               2021   GNN         Offline      Supervised     [#Cui2021mvgan]_
PP-GCN              2021   GNN         Online       Supervised     [#Peng2021ppgcn]_
KPGNN               2021   GNN         Online       Supervised     [#Cao2021kpgnn]_
FinEvent            2022   GNN         Online       Supervised     [#Peng2022finevent]_
QSGNN               2022   GNN         Online       Supervised     [#Ren2022qsgnn]_
ETGNN               2023   GNN         Offline      Supervised     [#Ren2023etgnn]_
HCRC                2023   GNN         Online       Unsupervised   [#Guo2023hcrc]_
UCLSED              2023   GNN         Offline      Supervised     [#Ren2023uclsad]_
RPLMSED             2024   PLM         Online       Supervised     [#Li2024rplmsed]_
HISEvent            2024   Others      Online       Unsupervised   [#Cao2024hisevent]_
==================  =====  ==========  ==========  ============  =====================



Collected Datasets
------------------

-   **ACE2005**: The ACE2005 dataset is a comprehensive collection of news articles annotated for entities, relations, and events. It includes a diverse range of event types and is widely used for event extraction research.
-   **MAVEN**: MAVEN (Massive event) is a large-scale dataset for event detection that consists of over 11,000 events annotated across a wide variety of domains. It is designed to facilitate the development of robust event detection models.
-   **TAC KBP**: The TAC KBP dataset is part of the Text Analysis Conference Knowledge Base Population track. It contains annotated events, entities, and relations, focusing on extracting structured information from unstructured text.
-   **CrisisLexT26**: CrisisLexT26 is a dataset containing tweets related to 26 different crisis events. It is used to study information dissemination and event detection in social media during emergencies.
-   **CrisisLexT6**: CrisisLexT6 is a smaller dataset from the CrisisLex collection, focusing on six major crisis events. It includes annotated tweets that provide valuable insights into public response and information spread during crises.
-   **Event2012**: Event2012 is a dataset composed of tweets related to various events in 2012. It includes a wide range of event types and is used for studying event detection and classification in social media.
-   **Event2018**: Event2018 consists of French tweets annotated for different event types. It provides a multilingual perspective on event detection, allowing researchers to explore language-specific challenges and solutions.
-   **KBP2017**: KBP2017 is part of the Knowledge Base Population track and focuses on extracting entities, relations, and events from text. It is a valuable resource for developing and benchmarking information extraction systems.
-   **CySecED**: CySecED is a dataset designed for cybersecurity event detection. It includes annotated cybersecurity events and is used to study threat detection and response in textual data.
-   **FewED**: FewED is a dataset for few-shot event detection, providing a limited number of annotated examples for each event type. It is designed to test the ability of models to generalize from few examples.


We provide their statistics as follows.

====================  ========  ==============  ==========  ==========  ==========
Dataset               Events    Event Types     Sentences   Tokens      Documents
====================  ========  ==============  ==========  ==========  ==========
ACE2005               5,349     33              11,738      230,382     599
MAVEN                 11,191    168             23,663      512,394     4,480
TAC KBP               3,500     18              7,800       150,000     2,500
CrisisLexT26          4,353     26              8,000       175,000     1,200
CrisisLexT6           2,100     6               4,500       90,000      600
Event2012             68,841    20              150,000     3,000,000   10,000
Event2018             15,000    10              50,000      1,000,000   5,000
KBP2017               4,200     22              9,000       180,000     3,000
CySecED               5,500     35              12,000      250,000     4,200
FewED                 6,000     40              14,000      300,000     5,500
====================  ========  ==============  ==========  ==========  ==========




API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Full API Reference: (https://docs.SocialED.org). API cheatsheet for all detectors:

* **preprocess()**\ :  Preprocess the dataset.
* **fit()**\ : Fit the detector with train data.
* **detector()**\: Initialize and configure the detection model, preparing it for training and prediction tasks.
* **evaluate(predictions, groundtruth)**\: Assess the performance of the detector by comparing predictions with the actual data.



How to Contribute
-----------------

You are welcome to become part of this project.
See `contribution guide <https://github.com/pygod-team/pygod/blob/main/CONTRIBUTING.rst>`_ for more information.





Contact
-------
Reach out to us by submitting an issue report or sending an email to sy2339225@buaa.edu.



References
----------
.. [#Blei2003lda] Blei, D.M., Ng, A.Y., and Jordan, M.I., 2003. Latent Dirichlet allocation. Journal of Machine Learning Research, 3(Jan), pp. 993-1022.

.. [#Graves2005bilstm] Graves, A., and Schmidhuber, J., 2005. Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural Networks, 18(5-6), pp. 602-610. Elsevier.

.. [#Mikolov2013word2vec] Mikolov, T., Chen, K., Corrado, G., and Dean, J., 2013. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

.. [#Pennington2014glove] Pennington, J., Socher, R., and Manning, C.D., 2014. GloVe: Global Vectors for Word Representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1532-1543. Association for Computational Linguistics.

.. [#Kusner2015wmd] Kusner, M., Sun, Y., Kolkin, N., and Weinberger, K., 2015. From word embeddings to document distances. In International Conference on Machine Learning, pp. 957-966. PMLR.

.. [#Devlin2018bert] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K., 2018. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

.. [#Reimers2019sbert] Reimers, N., and Gurevych, I., 2019. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3980-3990. Association for Computational Linguistics.

.. [#Liu2020eventx] Liu, B., Han, F.X., Niu, D., Kong, L., Lai, K., and Xu, Y., 2020. Story forest: Extracting events and telling stories from breaking news. ACM Transactions on Knowledge Discovery from Data (TKDD), 14(3), pp. 1-28. ACM New York, NY, USA.

.. [#Ren2021clkd] Ren, J., Peng, H., Jiang, L., Wu, J., Tong, Y., Wang, L., Bai, X., Wang, B., and Yang, Q., 2021. Transferring knowledge distillation for multilingual social event detection. arXiv preprint arXiv:2108.03084.

.. [#Cui2021mvgan] Cui, W., Zhang, Y., Liu, Z., and Yu, P.S., 2021. MVGAN: A Multi-view Graph Generative Adversarial Network for Anomaly Detection. In Proceedings of the 2021 IEEE International Conference on Big Data (Big Data), pp. 4513-4522. IEEE.

.. [#Peng2021ppgcn] Peng, H., Wu, J., Cao, Y., Dou, Y., Li, J., and Yu, P.S., 2021. PP-GCN: Privacy-Preserving Graph Convolutional Networks for Social Event Detection. In Proceedings of the Web Conference 2021, pp. 3383-3395.

.. [#Cao2021kpgnn] Cao, Y., Peng, H., Wu, J., Dou, Y., Li, J., and Yu, P.S., 2021. Knowledge-preserving incremental social event detection via heterogeneous GNNs. In Proceedings of the Web Conference 2021, pp. 3383-3395.

.. [#Peng2022finevent] Peng, H., Li, J., Gong, Q., Song, Y., Ning, Y., Lai, K., and Yu, P.S., 2019. Fine-grained event categorization with heterogeneous graph convolutional networks. arXiv preprint arXiv:1906.04580.

.. [#Ren2022qsgnn] Ren, J., Jiang, L., Peng, H., Cao, Y., Wu, J., Yu, P.S., and He, L., 2022. From known to unknown: Quality-aware self-improving graph neural network for open set social event detection. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, pp. 1696-1705.

.. [#Ren2023etgnn] Ren, J., Jiang, L., Peng, H., Liu, Z., Wu, J., and Yu, P.S., 2022. Evidential temporal-aware graph-based social event detection via Dempster-Shafer theory. In 2022 IEEE International Conference on Web Services (ICWS), pp. 331-336. IEEE.

.. [#Guo2023hcrc] Guo, Y., Zang, Z., Gao, H., Xu, X., Wang, R., Liu, L., and Li, J., 2024. Unsupervised social event detection via hybrid graph contrastive learning and reinforced incremental clustering. Knowledge-Based Systems, 284, p. 111225. Elsevier.

.. [#Ren2023uclsad] Ren, J., Jiang, L., Peng, H., Liu, Z., Wu, J., and Yu, P.S., 2023. Uncertainty-guided boundary learning for imbalanced social event detection. IEEE Transactions on Knowledge and Data Engineering. IEEE.

.. [#Li2024rplmsed] Li, P., Yu, X., Peng, H., Xian, Y., Wang, L., Sun, L., Zhang, J., and Yu, P.S., 2024. Relational Prompt-based Pre-trained Language Models for Social Event Detection. arXiv preprint arXiv:2404.08263.

.. [#Cao2024hisevent] Cao, Y., Peng, H., Yu, Z., and Philip, S.Y., 2024. Hierarchical and incremental structural entropy minimization for unsupervised social event detection. In Proceedings of the AAAI Conference on Artificial Intelligence, 38(8), pp. 8255-8264.

.. [#liu2024pygod] Liu, K., Dou, Y., Ding, X., Hu, X., Zhang, R., Peng, H., Sun, L., and Yu, P.S., 2024. PyGOD: A Python library for graph outlier detection. Journal of Machine Learning Research, 25(141), pp. 1-9.

.. [#zhao2019pyod] Zhao, Y., Nasrullah, Z., and Li, Z., 2019. PyOD: A python toolbox for scalable outlier detection. Journal of Machine Learning Research, 20(96), pp. 1-7.

.. [#wang2020maven] Wang, X., Wang, Z., Han, X., Jiang, W., Han, R., Liu, Z., Li, J., Li, P., Lin, Y., and Zhou, J., 2020. MAVEN: A massive general domain event detection dataset. arXiv preprint arXiv:2004.13590.

.. [#mcminn2013event2012] McMinn, A.J., Moshfeghi, Y., and Jose, J.M., 2013. Building a large-scale corpus for evaluating event detection on Twitter. In Proceedings of the 22nd ACM International Conference on Information & Knowledge Management, pp. 409-418.

.. [#mazoyer2020event2018] Mazoyer, B., Cagé, J., Hervé, N., and Hudelot, C., 2020. A French corpus for event detection on Twitter. European Language Resources Association (ELRA).
