# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import find_packages, setup


def parse_requirements(filename):
    with open(filename, "r") as file:
        return file.read().splitlines()


setup(
    name="synth_gen",
    version="0.1.0",
    author="Meta, FAIR, Olivier Duchenne",
    author_email="olivier1@meta.com",
    description="An open-source library designed for LLM to learn without Human supervision through execution feedback and more.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/synth_gen",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=parse_requirements("requirements.txt"),
)
