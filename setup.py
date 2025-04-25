from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="completion_tree_view",
    version="0.1.0",
    author="Brendan Hogan",
    author_email="brendan@bhogan.net",
    description="A library for visualizing token trees from language model completions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brendan-bhogan/CompletionTreeView",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "transformers>=4.20.0",
        "torch>=1.10.0",
        "graphviz>=0.17.0",
    ],
) 