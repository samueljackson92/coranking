try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os.path
import coranking

config = {
    'description': 'Co-ranking matricies for Python',
    'author': 'Samuel Jackson',
    'url': 'http://github.com/samueljackson92/coranking',
    'download_url': 'http://github.com/samueljackson92/coranking',
    'author_email': 'samueljackson@outlook.com',
    'version': coranking.__version__,
    'install_requires': [
	    'numpy',
	    'scipy',
	    'matplotlib',
	    'scikit-learn'
    ],
    'packages': ['coranking'],
    'name': 'coranking'
}

setup(**config)
