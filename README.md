# Co-ranking

[![Build Status](https://travis-ci.org/samueljackson92/coranking.svg?branch=master)](https://travis-ci.org/samueljackson92/coranking)
[![Documentation Status](https://readthedocs.org/projects/coranking/badge/?version=latest)](https://coranking.readthedocs.io/en/latest/?badge=latest)

Python implementation of a co-ranking matrix and various metrics derived from it.

Based on the methods discussed in:

> Lee, John A., and Michel Verleysen. "Quality assessment of dimensionality reduction: Rank-based criteria." Neurocomputing 72.7 (2009): 1431-1443.

Installation
-------------

It should be as simple as:

```python
pip install git+https://github.com/samueljackson92/coranking.git
```

This library has a module compiled with Cython. If you do not have Cython installed then the library will attempt to use the pre-generated C code included in this repo. If you find that this does not work consider installing Cython and reinstalling the library.
