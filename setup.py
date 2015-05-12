try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os.path
import coranking

path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, 'requirements.txt')

with open(path, 'r') as file_handle:
    requirements = file_handle.readlines()

config = {
    'description': 'Co-ranking matricies for Python',
    'author': 'Samuel Jackson',
    'url': 'http://github.com/samueljackson92/coranking',
    'download_url': 'http://github.com/samueljackson92/coranking',
    'author_email': 'samueljackson@outlook.com',
    'version': coranking.__version__,
    'install_requires': requirements,
    'packages': ['coranking'],
    'name': 'coranking'
}

setup(**config)
