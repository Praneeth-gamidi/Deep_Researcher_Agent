"""
Setup script for Deep Researcher Agent.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deep-researcher-agent",
    version="1.0.0",
    author="Deep Researcher Team",
    author_email="team@deepresearcher.ai",
    description="A comprehensive research agent with local embedding capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/deepresearcher/agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "web": ["streamlit>=1.28.0", "gradio>=3.50.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0", "black>=23.0.0", "flake8>=6.0.0"],
        "docs": ["sphinx>=5.0.0", "sphinx-rtd-theme>=1.2.0"],
    },
    entry_points={
        "console_scripts": [
            "deep-researcher=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "deep_researcher": ["*.py", "core/*.py", "storage/*.py", "utils/*.py", "interfaces/*.py"],
    },
    keywords="ai, research, embeddings, machine-learning, nlp, knowledge-base, vector-search",
    project_urls={
        "Bug Reports": "https://github.com/deepresearcher/agent/issues",
        "Source": "https://github.com/deepresearcher/agent",
        "Documentation": "https://deepresearcher.readthedocs.io/",
    },
)
