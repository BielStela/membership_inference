from setuptools import setup


def readme():
    with open("README.md") as readme_file:
        return readme_file.read()


configuration = {
    "name": "member-learn",
    "version": "0.0.1",
    "description": "Membership inference attacks with sklearn",
    "long_description": readme(),
    "classifiers": [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
    ],
    "keywords": "membership inference adversarial attack privacy machine-learning",
    "url": "https://github.com/BielStela/membership_inference",
    "maintainer": "Biel Stela",
    "maintainer_email": "biel.stela@gmail.com",
    "license": "BSD",
    "packages": ["mblearn"],
    "install_requires": [
        "numpy >= 1.13",
        "scikit-learn >= 0.16",
        "scipy >= 0.19",
        "pandas",
        "tqdm",
    ],
    "ext_modules": [],
    "cmdclass": {},
    "data_files": (),
}

setup(**configuration)
