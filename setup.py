from setuptools import find_packages, setup


setup(
    name="thomas",
    version="0.1",
    description="Tenstorrent Python Thomas",
    packages=["thomas"],
    package_dir={"thomas": "thomas"},
)