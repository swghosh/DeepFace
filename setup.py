import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepFace",
    version="1.0.0",
    author="Swarup Ghosh",
    author_email="codecrafts.cf@icloud.com",
    description='Keras implementation of the renowned publication "DeepFace: Closing the Gap to Human-Level Performance in Face Verification" by Taigman et al. Pre-trained weights on VGGFace2 dataset.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/swghosh/DeepFace",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Information Technology",
        "Topic :: Software Development :: Libraries :: Application Frameworks"
    ],
    python_requires='>=3.5',
)