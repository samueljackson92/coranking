try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import numpy as np
from Cython.Distutils import build_ext
from distutils.core import Extension

# define the extension module
ext_modules = [Extension("coranking._metrics_cy", ["src/_metrics_cy.pyx"])]
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
        'scikit-learn'
    ],
    'include_dirs': [np.get_include()],
    'ext_modules': ext_modules,
    'cmdclass': {'build_ext': build_ext},
    'packages': ['coranking'],
    'name': 'coranking'
}

setup(**config)
