from setuptools import setup, find_packages

requirements = [
    'torch==1.13.1',
    'torchvision==0.14.1',
    'transformers==4.27.4',
    'sentence-transformers==2.2.2',
    'accelerate==0.18.0',
    'torch-geometric==2.3.0',
    'dgl',
    'faiss-cpu',
    'numpy',
    'pandas',
    'scikit-learn',
    'networkx',
    'spacy',
    'gensim',
    'matplotlib',
    'seaborn',
]

setup(
    name='SocialED',
    version='1.1.2',
    packages=find_packages(),
    author='beici',
    author_email='zhangkun23@buaa.edu.cn',
    description='A Python Library for Social Event Detection',
    install_requires=requirements,
    include_package_data=True,
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RingBDStack/SocialED',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
