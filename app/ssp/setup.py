from setuptools import setup, find_packages

pkg_name = 'kfac_torch'


install_requires = [
    "torch-geometric==2.3.1",
    "torch-scatter==2.1.1",
    "torch>=1.13.1",
]

setup(
    name=pkg_name,
    version='0.0.1',
    install_requires=install_requires,
    packages=find_packages(where=pkg_name),
    url='',
    license='',
    author='simon',
    author_email='',
    description=''
)
