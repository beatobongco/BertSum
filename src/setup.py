import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bertsum",
    version="0.0.1",
    author="@nlpyang (packaged by @beatobongco)",
    description="BERT for text summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nlpyang/BertSum",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License 2.0",
    ],
)
