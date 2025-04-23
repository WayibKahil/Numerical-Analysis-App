from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="numerical-analysis-app",
    version="1.0.0",
    author="Hosam Dyab",
    author_email="hosamcode71@gmail.com",
    description="A modern desktop application for numerical analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hosamdyab/Numerical-Analysis-App",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "numerical-analysis-app=main:main",
        ],
    },
) 