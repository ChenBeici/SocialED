![SocialED Logo](https://github.com/RingBDStack/SocialED/blob/main/docs/SocialED.png?raw=true)

[![PyPI version](https://img.shields.io/pypi/v/socialed.svg?color=brightgreen)](https://pypi.org/project/SocialED/)
[![Documentation status](https://readthedocs.org/projects/socialed/badge/?version=latest)](https://socialed.readthedocs.io/en/latest/?badge=latest)
[![GitHub stars](https://img.shields.io/github/stars/RingBDStack/SocialED?style=flat)](https://github.com/RingBDStack/SocialED/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/RingBDStack/SocialED?style=flat)](https://github.com/RingBDStack/SocialED/network)
[![PyPI downloads](https://static.pepy.tech/personalized-badge/SocialED?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/SocialED)
[![testing](https://github.com/ChenBeici/SocialED/actions/workflows/testing.yml/badge.svg)](https://github.com/ChenBeici/SocialED/actions/workflows/testing.yml)
[![Coverage Status](https://coveralls.io/repos/github/pygod-team/pygod/badge.svg?branch=main)](https://coveralls.io/github/pygod-team/pygod?branch=main)
[![License](https://img.shields.io/github/license/RingBDStack/SocialED.svg)](https://github.com/RingBDStack/SocialED/blob/master/LICENSE)
[![CodeQL](https://github.com/RingBDStack/SocialED/actions/workflows/codeql.yml/badge.svg)](https://github.com/RingBDStack/SocialED/actions/workflows/codeql.yml)

---

# SocialED

A Python Library for Social Event Detection

## What is Social Event Detection?

Social Event Detection (SED) is a cutting-edge research area in AI and NLP that focuses on:

- 🔍 Automatically identifying and analyzing events from social media platforms (Twitter, Facebook, etc.)
- 🌎 Covering diverse event types from natural disasters to viral phenomena
- 🤖 Leveraging AI to understand real-world events through social data

## About SocialED

SocialED is your all-in-one Python toolkit for Social Event Detection that offers:

### 📊 Rich Resources
- 19 detection algorithms
- 14 diverse datasets
- Unified API with detailed documentation

### 🛠️ Key Capabilities
- Comprehensive preprocessing (graph construction, tokenization)
- Standardized interfaces for training & prediction
- Easy-to-extend modular architecture

### ⚡ Technical Excellence
- Deep learning framework integration
- CPU & GPU support for high performance
- Production-grade code quality with testing & CI/CD

## ⭐ Key Features

- **🤖 Comprehensive Algorithm Collection**: Integrates 19 detection algorithms and supports 14 widely-used datasets, with continuous updates to include emerging methods
- **📝 Unified API Design**: Implements algorithms with a consistent interface, allowing seamless data preparation and integration across all models
- **🔧 Modular Components**: Provides customizable components for each algorithm, enabling users to adjust models to specific needs
- **🛠️ Rich Utility Functions**: Offers tools designed to simplify the construction of social event detection workflows
- **✨ Robust Implementation**: Includes comprehensive documentation, examples, unit tests, and maintainability features

SocialED includes **19** social event detection algorithms.
For consistency and accessibility, SocialED is developed on top of [DGL](https://www.dgl.ai/) 
and [PyTorch](https://pytorch.org/), and follows the API design of [PyOD](https://github.com/yzhao062/pyod)
and [PyGOD](https://github.com/pygod-team/pygod).
See examples below for detecting outliers with SocialED in 7 lines!

SocialED plays a crucial role in various downstream applications, including:

* Crisis management
* Public opinion monitoring
* Fake news detection
* And more...

![SocialED API](https://github.com/RingBDStack/SocialED/blob/main/docs/API.png?raw=true)



## 📁 Folder Structure

|    Dataset    |       Language        |       Events        |       Texts        |    Long tail    |
| :-----------: | :---------------: | :------------------: | :------------------: | :---------------: |
| Event2012 | English | 503 | 68,841 | No |
| Event2018 | French | 257 | 64,516 | No |
| Arabic_Twitter | Arabic | 7 | 9,070 | No |
| MAVEN | English | 164 | 10,242 | No |
| CrisisLexT26 | English | 26 | 27,933 | No |
| CrisisLexT6 | English | 6 | 60,082 | No |
| CrisisMMD | English | 7 | 18,082 | No |
| CrisisNLP | English | 11 | 25,976 | No |
| HumAID | English | 19 | 76,484 | No |
| Mix_Data | English | 5 | 78,489 | No |
| KBP | English | 100 | 85,569 | No |
| Event2012_100 | English | 100 | 15,019 | Yes |
| Event2018_100 | French | 100 | 19,944 | Yes |
| Arabic_7 | Arabic | 7 | 3,022 | Yes |




## 🏗️ Library Design and Implementation

### 🔧 Dependencies and Technology Stack

SocialED is compatible with Python 3.8 and above, and leverages well-established deep learning frameworks like PyTorch and Hugging Face Transformers for efficient model training and inference, supporting both CPU and GPU environments. In addition to these core frameworks, SocialED also integrates NumPy, SciPy, and scikit-learn for data manipulation, numerical operations, and machine learning tasks, ensuring versatility and performance across a range of workflows.

### 🔄 Unified API Design

Inspired by the API designs of established frameworks, we developed a unified API for all detection algorithms in SocialED:

1. ``preprocess`` provides a flexible framework for handling various preprocessing tasks, such as graph construction and tokenization
2. ``fit`` trains the detection algorithms on the preprocessed data, adjusting model parameters and generating necessary statistics for predictions
3. ``detection`` uses the trained model to identify events from the input data, returning the detected events
4. ``evaluate`` assesses the performance of the detection results by comparing predictions with ground truth data, providing metrics like precision, recall and F1-score

### 💻 Example Usage

.. code-block:: python

    from SocialED.dataset import MAVEN                 # Load the dataset
    dataset = MAVEN().load_data()   # Load "arabic_twitter" dataset
    
    from SocialED.detector import KPGNN        # Import KPGNN model
    args = args_define().args                  # Get training arguments
    kpgnn = KPGNN(args, dataset)              # Initialize KPGNN model
    
    kpgnn.preprocess()                        # Preprocess data
    kpgnn.fit()                               # Train the model
    pres, trus = kpgnn.detection()            # Detect events
    kpgnn.evaluate(pres, trus)                # Evaluate detection results

### 🧩 Modular Design and Utility Functions

SocialED is built with a modular design to improve reusability and reduce redundancy. It organizes social event detection into distinct modules:

* ``preprocessing``
* ``modeling``
* ``evaluation``


The library provides several utility functions including:

* ``utils.tokenize_text`` and ``utils.construct_graph`` for data preprocessing
* ``metric`` for evaluation metrics
* ``utils.load_data`` for built-in datasets

## 🛡️ Library Robustness and Accessibility

### ✅ Quality and Reliability

* Built with robustness and high-quality standards
* Continuous integration through GitHub Actions
* Automated testing across Python versions and operating systems
* >99% code coverage
* PyPI-compatible and PEP 625 compliant
* Follows PEP 8 style guide

### 🤝 Accessibility and Community Support

* Detailed API documentation on Read the Docs
* Step-by-step guides and tutorials
* Intuitive API design inspired by scikit-learn
* Open-source project hosted on GitHub
* Easy issue-reporting mechanism
* Clear contribution guidelines

## 🔮 Future Development Plans

1. **Expanding Algorithms and Datasets**
   * Integrating advanced algorithms
   * Expanding datasets across languages, fields, and cultures

2. **Enhancing Intelligent Functions**
   * Automated machine learning for model selection
   * Hyperparameter optimization

3. **Supporting Real-time Detection**
   * Enhanced real-time event detection
   * Trend analysis capabilities
   * Support for streaming data


@misc{zhang2024socialedpythonlibrarysocial,
      title={SocialED: A Python Library for Social Event Detection}, 
      author={Kun Zhang and Xiaoyan Yu and Pu Li and Hao Peng and Philip S. Yu},
      year={2024},
      eprint={2412.13472},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.13472}, 
}


## 📊 Citation

```bibtex
@misc{zhang2024socialedpythonlibrarysocial,
      title={SocialED: A Python Library for Social Event Detection}, 
      author={Kun Zhang and Xiaoyan Yu and Pu Li and Hao Peng and Philip S. Yu},
      year={2024},
      eprint={2412.13472},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.13472}, 
}
```



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

.. [#Cao2021kpgnn] Cao, Y., Peng, H., Wu, J., Dou, Y., Li, J., and Yu, P.S., 2021. Knowledge-preserving incremental social event detection via heterogeneous GNNs. In Proceedings of the Web Conference 2021, pp. 3383-3395.

.. [#Peng2022finevent] Peng, H., Li, J., Gong, Q., Song, Y., Ning, Y., Lai, K., and Yu, P.S., 2019. Fine-grained event categorization with heterogeneous graph convolutional networks. arXiv preprint arXiv:1906.04580.

.. [#Ren2022qsgnn] Ren, J., Jiang, L., Peng, H., Cao, Y., Wu, J., Yu, P.S., and He, L., 2022. From known to unknown: Quality-aware self-improving graph neural network for open set social event detection. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, pp. 1696-1705.

.. [#Ren2023etgnn] Ren, J., Jiang, L., Peng, H., Liu, Z., Wu, J., and Yu, P.S., 2022. Evidential temporal-aware graph-based social event detection via Dempster-Shafer theory. In 2022 IEEE International Conference on Web Services (ICWS), pp. 331-336. IEEE.

.. [#Guo2023hcrc] Guo, Y., Zang, Z., Gao, H., Xu, X., Wang, R., Liu, L., and Li, J., 2024. Unsupervised social event detection via hybrid graph contrastive learning and reinforced incremental clustering. Knowledge-Based Systems, 284, p. 111225. Elsevier.

.. [#Ren2023uclsad] Ren, J., Jiang, L., Peng, H., Liu, Z., Wu, J., and Yu, P.S., 2023. Uncertainty-guided boundary learning for imbalanced social event detection. IEEE Transactions on Knowledge and Data Engineering. IEEE.

.. [#Li2024rplmsed] Li, P., Yu, X., Peng, H., Xian, Y., Wang, L., Sun, L., Zhang, J., and Yu, P.S., 2024. Relational Prompt-based Pre-trained Language Models for Social Event Detection. arXiv preprint arXiv:2404.08263.

.. [#Cao2024hisevent] Cao, Y., Peng, H., Yu, Z., and Philip, S.Y., 2024. Hierarchical and incremental structural entropy minimization for unsupervised social event detection. In Proceedings of the AAAI Conference on Artificial Intelligence, 38(8), pp. 8255-8264.

.. [#Yang2024adpsemevent] Yang, Z., Wei, Y., Li, H., et al. Adaptive Differentially Private Structural Entropy Minimization for Unsupervised Social Event Detection[C]//Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 2024: 2950-2960.

.. [#liu2024pygod] Liu, K., Dou, Y., Ding, X., Hu, X., Zhang, R., Peng, H., Sun, L., and Yu, P.S., 2024. PyGOD: A Python library for graph outlier detection. Journal of Machine Learning Research, 25(141), pp. 1-9.

.. [#zhao2019pyod] Zhao, Y., Nasrullah, Z., and Li, Z., 2019. PyOD: A python toolbox for scalable outlier detection. Journal of Machine Learning Research, 20(96), pp. 1-7.

.. [#wang2020maven] Wang, X., Wang, Z., Han, X., Jiang, W., Han, R., Liu, Z., Li, J., Li, P., Lin, Y., and Zhou, J., 2020. MAVEN: A massive general domain event detection dataset. arXiv preprint arXiv:2004.13590.

.. [#mcminn2013event2012] McMinn, A.J., Moshfeghi, Y., and Jose, J.M., 2013. Building a large-scale corpus for evaluating event detection on Twitter. In Proceedings of the 22nd ACM International Conference on Information & Knowledge Management, pp. 409-418.

.. [#mazoyer2020event2018] Mazoyer, B., Cagé, J., Hervé, N., and Hudelot, C., 2020. A French corpus for event detection on Twitter. European Language Resources Association (ELRA).



