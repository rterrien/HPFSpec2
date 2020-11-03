#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:13:04 2019

@author: rterrien
"""

import setuptools
#from numpy.distutils.core import setup, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="hpfspec2", # Replace with your own username
    version="0.0.2",
    author="RCT,GKS,KO,AK,FO,AI",
    author_email="rterrien@carleton.edu",
    description="HPFspec2 Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
#    url="https://github.com/pypa/sampleproject",
    #packages=setuptools.find_packages(where='./src'),
    package_dir={"hpfspec2":"src"},
#    include_package_data=True,
    packages=setuptools.find_packages(),#[''],#setuptools.find_packages(),
    py_modules=['hpfspec2.bary',
                'hpfspec2.rv_utils',
                'hpfspec2.spec_help',
                'hpfspec2.target',
                'hpfspec2.utils',
                'hpfspec2.hpfspec2',
                'hpfspec2.vsini_utils',
                'hpfspec2.model_utils',
                'hpfspec2.fitting_utils',
                'hpfspec2.calibration_utils'],
    #py_modules=['ccf.mask'],
    #packages=['mask'],
#    classifiers=[
#        "Programming Language :: Python :: 2",
#        "License :: OSI Approved :: MIT License",
#        "Operating System :: OS Independent",
#    ],
    python_requires='>=2.7',
)
