from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from package init
def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "glcm_analyzer", "__init__.py"), "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="glcm_analyzer",
    version=get_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="A high-performance GLCM texture analysis package for satellite imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/glcm_analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Natural Language :: English",
    ],
    python_requires=">=3.7",
    install_requires=[
        "rasterio>=1.3.9",      
        "numpy>=1.24.0",        
        "scikit-image>=0.21.0", 
        "matplotlib>=3.7.0",    
        "pandas>=2.1.0",        
        "tqdm>=4.66.0",         
        "geopandas>=0.14.0",    
        "shapely>=2.0.0",       
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
        "full": [
            "dask>=2023.0.0",
            "scipy>=1.11.0",
            "seaborn>=0.13.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "performance": [
            "numba>=0.58.0",
            "dask>=2023.0.0",
            "zarr>=2.16.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "glcm-optimize=glcm_analyzer.cli:optimize_cli",
            "glcm-process=glcm_analyzer.cli:process_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "glcm_analyzer": [
            "examples/*.py",
            "data/*.json",
        ],
    },
    keywords=[
        "remote-sensing",
        "gis",
        "texture-analysis",
        "glcm",
        "satellite-imagery",
        "image-processing",
        "geospatial",
        "raster-analysis",
        "python-3.13",
    ],
    project_urls={
        "Bug Reports": "https://github.com/DijoG/glcm_analyzer/issues",
        "Source": "https://github.com/DijoG/glcm_analyzer",
        "Documentation": "https://github.com/DijoG/glcm_analyzer/README.md",
    },
)