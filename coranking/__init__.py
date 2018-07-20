# We first need to detect if we're being called as part of the setup.py
try:
    __CORANKING_SETUP__
except NameError:
    __CORANKING_SETUP__ = False

if not __CORANKING_SETUP__:
    from coranking._coranking import coranking_matrix
    __all__ = ['coranking_matrix']

__version__ = "0.1.1"


