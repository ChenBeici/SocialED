Installation
============


It is recommended to use **pip** for installation.
Please make sure **the latest version** is installed, as SocialED is updated frequently:

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

**Required Dependencies**\ :

* python>=3.8
* numpy>=1.24.4
* scikit-learn>=1.3.2
* scipy>=1.10.1
* networkx>=3.1
* dgl>=0.6.0


**Note on PyG and PyTorch Installation**\ :
SocialED depends on `torch <https://https://pytorch.org/get-started/locally/>`_ and `torch_geometric (including its optional dependencies) <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#>`_.
To streamline the installation, SocialED does **NOT** install these libraries for you.
Please install them from the above links for running SocialED:


**Required Dependencies**\ :

* python>=3.8
* numpy>=1.24.3
* scikit-learn>=1.2.2
* scipy>=1.10.1
* networkx>=2.3
* torch>=2.3.0
* torch_geometric>=2.5.3
* dgl>=0.6.0