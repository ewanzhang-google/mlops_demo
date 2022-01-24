from setuptools import setup
import os, sys, pathlib

with open(os.path.dirname(pathlib.Path(__file__).absolute())+'/requirements.txt') as f:
    required = f.read().splitlines()

    
setup(
    name='MLOps Pipeline Example',
    version='0.1',
    packages=[],
    url='',
    license='',
    author='aniftos',
    author_email='aniftos@google.com',
    description='',
    install_requires=required
)