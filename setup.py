import sys
if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.core import Extension

try:
    # Try building with Cython
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    ext_modules = cythonize([Extension("coranking._metrics_cy",
                            ["src/_metrics_cy.pyx"])])
except ImportError:
    # Else just use the C file from the repo
    from distutils.command.build_ext import build_ext
    ext_modules = [Extension("coranking._metrics_cy", ["src/_metrics_cy.c"])]

builtins.__CORANKING_SETUP__ = True


class BuildCorankingExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):
        import numpy as np
        self.include_dirs.append(np.get_include())
        build_ext.run(self)

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
    ],
    'ext_modules': ext_modules,
    'cmdclass': {'build_ext': BuildCorankingExtCommand},
    'packages': ['coranking'],
    'name': 'coranking'
}

setup(**config)

del builtins.__CORANKING_SETUP__

