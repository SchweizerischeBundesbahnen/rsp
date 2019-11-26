from setuptools import setup, find_packages

setup(
    name='rsp',
    version='0.0.1',
    url='git@github.com:SchweizerischeBundesbahnen/rsp.git',
    author='Erik Nygren, Christian Eichenberger, Adrian Egli, Christian Baumberger',
    author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages(),
    # TODO requirements see tox.ini
    install_requires=[],
)
