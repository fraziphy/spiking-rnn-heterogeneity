# setup.py
"""
Setup configuration for Spiking RNN Heterogeneity Framework v3.3.0
Split experiments architecture: spontaneous activity + network stability analysis
NEW: Poisson process analysis, rate-dependent RNG, transient removal, fixed time-step bug
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
        return "Split experiments framework for studying spontaneous activity and network stability in heterogeneous spiking neural networks with Poisson process validation"

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
        # Fallback requirements for split experiments with Poisson analysis
        return [
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "mpi4py>=3.1.0",
            "psutil>=5.8.0",
            "matplotlib>=3.5.0",
        ]

setup(
    name="spiking-rnn-heterogeneity",
    version="3.3.0",

    description="Split experiments framework with Poisson process validation: spontaneous activity + network stability analysis for heterogeneous spiking networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",

    author="Computational Neuroscience Research Group",
    author_email="research@example.com",
    url="https://github.com/yourusername/spiking-rnn-heterogeneity",

    # Package discovery
    packages=find_packages(),

    # Include non-code files
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.sh'],
        'runners': ['*.sh'],
        'tests': ['*.py'],
        'results': ['*.pkl', '*.json'],
    },

    # Dependencies for split experiments with Poisson analysis
    python_requires=">=3.8",
    install_requires=read_requirements(),

    # Optional dependencies for enhanced features
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "analysis": [
            "pandas>=1.3.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0.0",
            "networkx>=2.6.0",
            "h5py>=3.0.0",
            "plotly>=5.0.0",
        ],
        "optimized": [
            # Additional packages for optimized computation
            "numba>=0.56.0",  # JIT compilation for fast LZ and coincidence computation
            "joblib>=1.1.0",  # Parallel processing for pattern analysis
            "tqdm>=4.62.0",   # Progress bars for long computations
        ],
        "statistics": [
            # Enhanced statistical analysis for Poisson validation
            "statsmodels>=0.13.0",  # Advanced statistical tests
            "pingouin>=0.5.0",      # Statistical functions
            "scipy>=1.9.0",         # Latest statistical functions
        ],
        "hpc": [
            # High-performance computing extensions
            "mpi4py>=3.1.0",
            "h5py>=3.0.0",
            "tables>=3.6.0",
        ],
        "all": [
            # All optional dependencies
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pandas>=1.3.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0.0",
            "networkx>=2.6.0",
            "h5py>=3.0.0",
            "plotly>=5.0.0",
            "numba>=0.56.0",
            "joblib>=1.1.0",
            "tqdm>=4.62.0",
            "tables>=3.6.0",
            "statsmodels>=0.13.0",
            "pingouin>=0.5.0",
        ]
    },

    # Entry points for split experiment command-line tools
    entry_points={
        'console_scripts': [
            'spiking-rnn-test=tests.test_installation:main',
            'spiking-rnn-structure-test=tests.test_comprehensive_structure:run_all_comprehensive_tests',
            'spiking-rnn-spontaneous=runners.mpi_spontaneous_runner:main',
            'spiking-rnn-stability=runners.mpi_stability_runner:main',
        ],
    },

    # Project classification for split experiments framework with Poisson analysis
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",

        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",

        # Topic - Split experiments categories with Poisson analysis
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",

        # License
        "License :: OSI Approved :: MIT License",

        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",

        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",

        # Environment
        "Environment :: Console",
        "Environment :: Other Environment",
        "Environment :: GPU",
    ],

    # Keywords for PyPI discovery - split experiments with Poisson focus
    keywords=[
        # Core split experiments concepts
        "spiking neural networks",
        "spontaneous activity analysis",
        "network stability analysis",
        "heterogeneity",
        "computational neuroscience",
        "split experiments",

        # NEW: Poisson process features
        "poisson process analysis",
        "inter-spike interval",
        "coefficient of variation",
        "fano factor",
        "exponential distribution",
        "statistical validation",
        "transient removal",

        # Spontaneous activity features
        "firing rate analysis",
        "dimensionality analysis",
        "silent neurons",
        "multi-bin analysis",
        "participation ratio",

        # Network stability features
        "perturbation analysis",
        "lempel-ziv complexity",
        "coincidence analysis",
        "kistler coincidence",
        "gamma coincidence",
        "pattern stability",
        "hamming distance",

        # Technical optimizations and fixes
        "optimized coincidence",
        "unified calculation",
        "enhanced connectivity",
        "rate-dependent randomization",
        "time-step dependent rng",
        "fixed poisson bug",
        "randomized jobs",
        "cpu load balancing",
        "mpi parallelization",

        # Research applications
        "chaos quantification",
        "neural complexity",
        "network dynamics",
        "spatiotemporal patterns",
        "synaptic dynamics",
        "brain dynamics",
        "neural statistics",
    ],

    # Project URLs for split experiments with Poisson analysis
    project_urls={
        "Documentation": "https://github.com/yourusername/spiking-rnn-heterogeneity/wiki",
        "Bug Reports": "https://github.com/yourusername/spiking-rnn-heterogeneity/issues",
        "Source Code": "https://github.com/yourusername/spiking-rnn-heterogeneity",
        "Contributing": "https://github.com/yourusername/spiking-rnn-heterogeneity/blob/main/CONTRIBUTING.md",
        "Changelog": "https://github.com/yourusername/spiking-rnn-heterogeneity/blob/main/CHANGELOG.md",
        "Research Paper": "https://doi.org/your-paper-doi",
        "Examples": "https://github.com/yourusername/spiking-rnn-heterogeneity/tree/main/examples",
        "Spontaneous Analysis Guide": "https://github.com/yourusername/spiking-rnn-heterogeneity/wiki/Spontaneous-Activity",
        "Stability Analysis Guide": "https://github.com/yourusername/spiking-rnn-heterogeneity/wiki/Network-Stability",
        "Poisson Analysis Guide": "https://github.com/yourusername/spiking-rnn-heterogeneity/wiki/Poisson-Process-Validation",
        "Background Jobs Guide": "https://github.com/yourusername/spiking-rnn-heterogeneity/wiki/Background-Jobs",
    },

    # Package metadata
    zip_safe=False,

    # Additional metadata for split experiments framework with Poisson validation
    platforms=["any"],
    license="MIT",
)
