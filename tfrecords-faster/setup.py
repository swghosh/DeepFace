import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tfrecords-faster",
    version="0.0.1",
    author="Swarup Ghosh",
    author_email="codecrafts.cf@icloud.com",
    description="Gear up TensorFlow TFRecords preparation from images using multiprocessing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/swghosh/tfrecords-faster",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Information Technology",
        "Topic :: Utilities"
    ],
    python_requires='>=3.5',
)