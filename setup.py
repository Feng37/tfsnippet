"""
TF Snippet
----------

TF Snippet is a set of utilities for writing and testing TensorFlow models.
These utilities are in an early development stage, and might be migrated to
a new dedicated project once they are mature enough.
"""
import ast
import codecs
import os
import re
import sys
from setuptools import setup, find_packages


_version_re = re.compile(r'__version__\s+=\s+(.*)')
_source_dir = os.path.split(os.path.abspath(__file__))[0]

if sys.version_info[0] == 2:
    def read_file(path):
        with open(path, 'rb') as f:
            return f.read()
else:
    def read_file(path):
        with codecs.open(path, 'rb', 'utf-8') as f:
            return f.read()

version = str(ast.literal_eval(_version_re.search(
    read_file(os.path.join(_source_dir, 'tfsnippet/__init__.py'))).group(1)))

install_requires = read_file(os.path.join(_source_dir, 'requirements.txt'))
install_requires = list(filter(
    lambda v: v and not v.startswith('#'),
    (s.strip() for s in install_requires.split('\n'))
))


setup(
    name='TFSnippet',
    version=version,
    url='https://github.com/korepwx/tfsnippet/',
    license='MIT',
    author='Haowen Xu',
    author_email='public@korepwx.com',
    description='A set of utilities for writing and testing TensorFlow models.',
    long_description=__doc__,
    packages=find_packages('.', exclude=['tests', 'tests.*']),
    zip_safe=False,
    platforms='any',
    setup_requires=['setuptools'],
    install_requires=install_requires,
    dependency_links=[
        'git+https://github.com/korepwx/mlcomp.git@dev'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
