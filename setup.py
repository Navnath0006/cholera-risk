"""
Setup script for the Cholera Risk Assessment package.

Install:
    pip install .

Install in development mode:
    pip install -e .

Run tests:
    python -m pytest tests/ -v
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cholera_risk",
    version="1.0.0",
    author="Navnath Kamble, Yamini Madugu",
    author_email="navnathkamble0007@gmail.com, maduguyamini63662@gmail.com",
    description="A Monte Carlo simulation for Cholera risk assessment — "
                "quantitative transboundary infection risk analysis with configurable parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Navnath0006/cholera-risk",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "cholera-risk=cholera_risk.model:quick_risk_assessment",
        ],
    },
)
