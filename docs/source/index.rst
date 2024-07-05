.. SocialED documentation master file, created by
   sphinx-quickstart on [创建日期].
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: ./_static/socialED.png
    :scale: 30%
    :alt: logo


----

SocialED is a **Python library** for **social event detection**.
This exciting yet challenging field has many key applications, e.g., detecting
suspicious activities in social networks and analyzing social media trends.

SocialED includes **multiple** social event detection algorithms.
For consistency and accessibility, SocialED is developed on top of `PyTorch <https://pytorch.org/>`_
and follows the API design of `scikit-learn <https://scikit-learn.org/>`.
See examples below for detecting social events with SocialED in a few lines of code!

**SocialED is featured for**:

* **Unified APIs, detailed documentation, and interactive examples** across various social event detection algorithms.
* **Comprehensive coverage** of various social event detectors.
* **Support for processing large datasets** via efficient data handling methods.
* **Seamless integration with PyTorch** for advanced deep learning applications.

**Event Detection Using SocialED with a Few Lines of Code**:

.. code-block:: python

    # train a BERT-based event detector
    from Socialed.detector import BERTModel

    model = BERTModel(args, dataset)  # initialize model with arguments and dataset
    model.fit()  # train the model

    # predict labels and scores on the testing data
    predictions = model.prediction()
    print(predictions)

----

Implemented Algorithms
----------------------

==================  =====  ===========  ===========  ==============================================
Abbr                Year   Backbone     Sampling     Class
==================  =====  ===========  ===========  ==============================================
BERT                2021   Transformer  Yes          :class:`Socialed.detector.BERTModel`
Event2012           2012   Classic ML   No           :class:`Socialed.detector.Event2012Model`
==================  =====  ===========  ===========  ==============================================

----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   tutorials/index
   api
   minibatch

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: API References

   socialed.detector
   socialed.dataset
   socialed.metrics
   socialed.test
   socialed.utils


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   cite
   team
   reference
