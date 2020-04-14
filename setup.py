import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read()

setuptools.setup(
    name="palladio",
    version=version,
    author="Matteo Barbieri",
    author_email="matteo.barbieri@peltarion.com",
    description="Classification toolkit (for NN)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matteo-peltarion/classification-toolkit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    scripts=[
        'bin/train.py',
    ]
)
