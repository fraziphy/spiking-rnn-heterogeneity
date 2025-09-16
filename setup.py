# setup.py
"""
Setup configuration for Spiking RNN Heterogeneity Studies Framework.
Enables cross-directory imports and package management.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Framework for studying chaos in heterogeneous spiking neural networks"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    except FileNotFoundError:
        # Fallback requirements if file doesn't exist
        return [
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "mpi4py>=3.1.0",
            "psutil>=5.8.0",
            "matplotlib>=3.3.0",
        ]

setup(
    name="spiking-rnn-heterogeneity",
    version="1.0.0",

    description="Framework for studying chaos in heterogeneous spiking neural networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",

    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/spiking-rnn-heterogeneity",

    # Package discovery
    packages=find_packages(),

    # Include non-code files
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.sh'],
        'runners': ['*.sh'],
    },

    # Dependencies
    python_requires=">=3.8",
    install_requires=read_requirements(),

    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "analysis": [
            "pandas>=1.3.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0.0",
            "networkx>=2.6.0",
            "h5py>=3.0.0",
        ],
        "all": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "pandas>=1.3.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0.0",
            "networkx>=2.6.0",
            "h5py>=3.0.0",
        ]
    },

    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'spiking-rnn-test=tests.test_installation:main',
        ],
    },

    # Project classification
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",

        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",

        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",

        # License
        "License :: OSI Approved :: MIT License",

        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",

        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",

        # Environment
        "Environment :: Console",
        "Environment :: Other Environment",
    ],

    # Keywords for PyPI
    keywords=[
        "spiking neural networks",
        "chaos theory",
        "heterogeneity",
        "computational neuroscience",
        "network dynamics",
        "lempel-ziv complexity",
        "mpi parallelization",
        "perturbation analysis",
    ],

    # Project URLs
    project_urls={
        "Documentation": "https://github.com/yourusername/spiking-rnn-heterogeneity/wiki",
        "Bug Reports": "https://github.com/yourusername/spiking-rnn-heterogeneity/issues",
        "Source Code": "https://github.com/yourusername/spiking-rnn-heterogeneity",
        "Contributing": "https://github.com/yourusername/spiking-rnn-heterogeneity/blob/main/CONTRIBUTING.md",
    },

    # Minimum Python version check
    zip_safe=False,
)
