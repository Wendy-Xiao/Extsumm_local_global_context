import os
from setuptools import setup

setup(
    name='rouge_papier_v2',
    version='0.0.1',
    description='A python ROUGE wrapper.',
    author='Chris Kedzie',
    author_email='kedzie@cs.columbia.edu',
    packages=["rouge_papier_v2"],
    dependency_links = [],
    install_requires = ["pandas"],
    package_data={
        '': [os.path.join("data", 'ROUGE-1.5.5.pl'),
             os.path.join("rouge_data", "*.txt"), 
             os.path.join("rouge_data", "*.db"), 
             os.path.join("rouge_data", "WordNet-1.6-Exceptions", "*.db"), 
             os.path.join("rouge_data", "WordNet-1.6-Exceptions", "*.pl"), 
             os.path.join("rouge_data", "WordNet-1.6-Exceptions", "*.exc"), 
             os.path.join("rouge_data", "WordNet-2.0-Exceptions", "*.db"), 
             os.path.join("rouge_data", "WordNet-2.0-Exceptions", "*.pl"), 
             os.path.join("rouge_data", "WordNet-2.0-Exceptions", "*.exc"), 
            ],
    }
)
