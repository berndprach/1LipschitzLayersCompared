from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
name='lipnn',
version='0.0.1',
author='Fabio Brau, Bernd Prach',
author_email='fabio.br.92@hotmail.it, bernd.prach@ist.ac.at',
description='This package offers a handy implementation of Lipschitz Bounded Layers for crafting Lipschitz Bounded Neural Networks.',
packages=find_packages(),
url='https://github.com/berndprach/1LipschitzLayersCompared',
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
python_requires='>=3.6',
)