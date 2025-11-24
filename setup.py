from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="trustworthy-ai",
    version="0.1.0",
    author="Équipe Trustworthy AI",
    author_email="team@trustworthy-ai.edu",
    description="Projet d'éthique en IA - Classification faciale avec apprentissage fédéré",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/team/trustworthy-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.9.0",
            "flake8>=3.9.0",
            "pre-commit>=2.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "trustworthy-ai=src.main:main",
        ],
    },
)