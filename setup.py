from setuptools import setup, find_packages

setup(
    name="drift-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "xgboost>=1.5.0",
        "ipywidgets>=7.6.0",
        "tensorflow>=2.8.0",
        "torch>=1.10.0",
        "river>=0.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for detecting concept drift in machine learning models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/drift-detection",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)