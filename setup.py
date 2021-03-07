#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# https://datasciencejob.de/articles/how-to-setup-your-python-data-science-projects-to-save-you-hassle-time-money
from __future__ import absolute_import
from __future__ import print_function

import io

from os.path import dirname
from os.path import join

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='replicability',
    version='0.1.0',
    license='MIT license',
    description='Towards Replicability',
    author='Davit Svanidze',
    author_email='svanskivar@gmail.com',
    url='https://github.com/dsvanidze',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    zip_safe=False,
)