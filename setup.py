from setuptools import setup, find_packages

setup(
    name="leafandloaf-ai",
    version="0.1.0",
    description="ML-powered recommendation system for grocery retail",
    author="LeafAndLoaf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "torch>=2.0.0",
        "google-cloud-aiplatform>=1.35.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-recommendations=train_recommendations:main",
            "generate-data=data_generator:main",
        ]
    },
)