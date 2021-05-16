import setuptools
from os import path

__version__ = "0.0.1"

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split('\n')

requirements_installation = [x.strip() for x in all_reqs if "git+" not in x]

requirements_installation.pop()

setuptools.setup(
    name="MachineLearning_Scratch",
    version=__version__,
    author="Jesse Khaira",
    author_email="jesse.khaira15@gmail.com",
    license="MIT",
    description=
    "Python implementations using only NumPy of some foundational ML algorithms",
    url="https://github.com/13jk59/MachineLearning_Scratch.git",
    packages=setuptools.find_packages(),
    install_requires=requirements_installation)
