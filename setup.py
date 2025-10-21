# setup.py
"""
Setup configuration for Spiking RNN Heterogeneity Framework v6.0.0

NEW in v6.0.0:
- Reservoir computing tasks: categorical classification, temporal transformation, auto-encoding
- Pattern-based HD input generation with task-specific caching
- Dimensionality analysis for auto-encoding experiments
- Unified TaskPerformanceExperiment infrastructure

Refactored in v5.1.0:
- Eliminated code duplication across experiments and analysis modules
- Base experiment class with shared methods
- Unified utilities in common_utils and experiment_utils
"""

from setuptools import setup, find_packages
import os

def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ("Framework for studying spontaneous activity, network stability, HD encoding, "
                "and reservoir computing tasks in heterogeneous spiking neural networks")

def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    except FileNotFoundError:
        return [
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "mpi4py>=3.1.0",
            "psutil>=5.8.0",
            "matplotlib>=3.5.0",
        ]

setup(
    name="spiking-rnn-heterogeneity",
    version="6.0.0",
    description="Reservoir computing tasks with refactored framework: categorical, temporal, and auto-encoding experiments",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Computational Neuroscience Research Group",
    author_email="research@example.com",
    url="https://github.com/yourusername/spiking-rnn-heterogeneity",

    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.sh'],
        'runners': ['*.sh', '*.py'],
        'tests': ['*.py'],
        'results': ['*.pkl', '*.json'],
        'hd_signals': ['*.pkl'],
    },

    python_requires=">=3.8",
    install_requires=read_requirements(),

    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "analysis": [
            "pandas>=1.3.0",
            "seaborn>=0.11.0",
            "networkx>=2.6.0",
            "h5py>=3.0.0",
            "plotly>=5.0.0",
        ],
        "all": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "black>=21.0.0",
            "pandas>=1.3.0",
            "seaborn>=0.11.0",
            "networkx>=2.6.0",
            "h5py>=3.0.0",
            "plotly>=5.0.0",
        ]
    },

    entry_points={
        'console_scripts': [
            # Testing
            'spiking-rnn-test=tests.test_installation:main',
            'spiking-rnn-structure-test=tests.test_comprehensive_structure:run_all_comprehensive_tests',
            'spiking-rnn-encoding-test=tests.test_encoding_implementation:main',
            'spiking-rnn-task-test=tests.test_task_performance:main',  # NEW in v6.0
            # Original experiments
            'spiking-rnn-spontaneous=runners.mpi_spontaneous_runner:main',
            'spiking-rnn-stability=runners.mpi_stability_runner:main',
            'spiking-rnn-encoding=runners.mpi_encoding_runner:main',
            # NEW in v6.0: Task experiments
            'spiking-rnn-task=runners.mpi_task_runner:main',
            'spiking-rnn-autoencoding=runners.mpi_autoencoding_runner:main',
        ],
    },

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],

    keywords=[
        "spiking neural networks",
        "heterogeneity",
        "reservoir computing",
        "categorical classification",
        "temporal transformation",
        "auto-encoding",
        "spontaneous activity",
        "network stability",
        "HD encoding",
        "dimensionality analysis",
        "refactored code",
        "computational neuroscience",
    ],

    project_urls={
        "Documentation": "https://github.com/yourusername/spiking-rnn-heterogeneity/wiki",
        "Bug Reports": "https://github.com/yourusername/spiking-rnn-heterogeneity/issues",
        "Source Code": "https://github.com/yourusername/spiking-rnn-heterogeneity",
        "Changelog": "https://github.com/yourusername/spiking-rnn-heterogeneity/releases",
    },

    zip_safe=False,
    platforms=["any"],
    license="MIT",
)
