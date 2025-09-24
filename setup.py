# setup.py
"""
Setup configuration for Spiking RNN Heterogeneity Framework v2.1.0
Enhanced complexity analysis with 4 LZ measures, Kistler coincidence, and pattern stability.
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
        return "Enhanced framework for studying chaos and complexity in heterogeneous spiking neural networks with comprehensive analysis tools"

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
        # Fallback requirements for enhanced analysis
        return [
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "mpi4py>=3.1.0",
            "psutil>=5.8.0",
            "matplotlib>=3.5.0",
        ]

setup(
    name="spiking-rnn-heterogeneity",
    version="2.1.0",

    description="Enhanced framework for chaos and complexity analysis in heterogeneous spiking neural networks",
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

    # Dependencies for enhanced analysis
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
        "enhanced": [
            # Additional packages for enhanced complexity analysis
            "numba>=0.56.0",  # JIT compilation for fast LZ computation
            "joblib>=1.1.0",  # Parallel processing for pattern analysis
            "tqdm>=4.62.0",   # Progress bars for long computations
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
        ]
    },

    # Entry points for enhanced command-line tools
    entry_points={
        'console_scripts': [
            'spiking-rnn-test=tests.test_installation:main',
            'spiking-rnn-structure-test=tests.test_comprehensive_structure:run_all_comprehensive_tests',
            'spiking-rnn-chaos=runners.mpi_chaos_runner:main',
        ],
    },

    # Project classification for enhanced framework
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",

        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",

        # Topic - Enhanced categories
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

    # Enhanced keywords for PyPI discovery
    keywords=[
        # Core concepts
        "spiking neural networks",
        "chaos theory",
        "heterogeneity",
        "computational neuroscience",
        "network dynamics",

        # Enhanced analysis features
        "lempel-ziv complexity",
        "perturbational complexity index",
        "pci",
        "kistler coincidence",
        "pattern stability",
        "dimensionality analysis",
        "participation ratio",

        # Technical features
        "mpi parallelization",
        "perturbation analysis",
        "synaptic dynamics",
        "firing rate analysis",
        "temporal precision",
        "multi-resolution analysis",

        # Research applications
        "chaos quantification",
        "neural complexity",
        "network topology",
        "spatiotemporal patterns",
        "consciousness research",
        "brain dynamics",
    ],

    # Enhanced project URLs
    project_urls={
        "Documentation": "https://github.com/yourusername/spiking-rnn-heterogeneity/wiki",
        "Bug Reports": "https://github.com/yourusername/spiking-rnn-heterogeneity/issues",
        "Source Code": "https://github.com/yourusername/spiking-rnn-heterogeneity",
        "Contributing": "https://github.com/yourusername/spiking-rnn-heterogeneity/blob/main/CONTRIBUTING.md",
        "Changelog": "https://github.com/yourusername/spiking-rnn-heterogeneity/blob/main/CHANGELOG.md",
        "Research Paper": "https://doi.org/your-paper-doi",
        "Examples": "https://github.com/yourusername/spiking-rnn-heterogeneity/tree/main/examples",
    },

    # Package metadata
    zip_safe=False,

    # Additional metadata for enhanced framework
    platforms=["any"],
    license="MIT",
)
