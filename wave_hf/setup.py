"""
Setup script for WAVe HuggingFace package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wave-hf",
    version="1.0.0",
    author="Yuriy Perezhohin, Mauro Castelli",
    author_email="yperezhohin@novaims.unl.pt",
    description="WAVe: Word-Aligned Verification of Synthetic Speech for ASR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuriyvnv/WAVe",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "examples": [
            "datasets>=2.0.0",
            "librosa>=0.10.0",
            "tqdm>=4.60.0",
            "matplotlib>=3.5.0",
        ],
    },
)
