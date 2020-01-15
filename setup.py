import os

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()


def get_all_files_in_directory(directory: str):
    ret = []
    for dirpath, subdirs, files in os.walk(directory):
        for f in files:
            ret.append(os.path.join(dirpath, f))
    return ret


setup(
    name='rsp',
    version='0.0.1',
    url='git@github.com:SchweizerischeBundesbahnen/rsp.git',
    author='Erik Nygren, Christian Eichenberger, Adrian Egli, Christian Baumberger',
    author_email='author@gmail.com',
    description='Description of my package',
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    # TODO requirements see tox.ini
    install_requires=[],
    include_package_data=True,
    data_files=[('res', get_all_files_in_directory('res')),
                ('tests', get_all_files_in_directory('tests'))],
    zip_safe=False,
)
