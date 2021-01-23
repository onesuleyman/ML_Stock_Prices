""" from setuptools import setup, find_packages

setup(
    name='funda_project',
    version='0.1.0',
    description='Example package for ML pipeline',
    packages=find_packages(include=['funda']),
    install_requires=[
        "pandas",
        "numpy",
        "plotnine",
        "scikit-learn",
        "seaborn",
        'tensorflow',
    ],
    python_requires="==3.8.5"
 """) 
