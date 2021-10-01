import os.path

from setuptools import find_packages, setup


def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    fp = os.path.join(here, "opti/__init__.py")
    for line in open(fp).readlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
    return ""


setup(
    name="basf-opti",
    description="Tools for experimental design and multi-objective optimization",
    url="https://gitlab.roqs.basf.net/bayesopt/opti",
    version=get_version(),
    packages=find_packages(),
    package_data={"": ["problems/data/*"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scipy>=1.7",
    ],
    extras_require={"testing": ["pytest", "scikit-learn"]},
    python_requires=">=3.6",
)
