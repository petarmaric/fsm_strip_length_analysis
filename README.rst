About
=====

Console app and Python API for strip length-dependent visualization and modal
analysis of the parametric model of buckling and free vibration in prismatic
shell structures, as computed by the `fsm_eigenvalue project`_.

This work is a part of the investigation within the research project
[ON174027]_, supported by the Ministry for Science and Technology, Republic of
Serbia. This support is gratefully acknowledged.

References
----------

.. [ON174027]
   "Computational Mechanics in Structural Engineering"

.. _`fsm_eigenvalue project`: https://github.com/petarmaric/fsm_eigenvalue

Installation
============

To install fsm_strip_length_analysis run::

    $ pip install fsm_strip_length_analysis

Console app usage
=================

Quick start::

    $ fsm_strip_length_analysis <filename>

Show help::

    $ fsm_strip_length_analysis --help

Python API usage
================

Quick start::

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)

    >>> from fsm_strip_length_analysis import analyze_model, configure_matplotlib

    >>> model_file = 'examples/barbero-viscoelastic.hdf5'
    >>> report_file = model_file.replace('.hdf5', '.pdf')

    >>> configure_matplotlib()
    >>> analyze_model(model_file, report_file, t_b_fix=6.25, add_automatic_markers=True)

Contribute
==========

If you find any bugs, or wish to propose new features `please let us know`_.

If you'd like to contribute, simply fork `the repository`_, commit your changes
and send a pull request. Make sure you add yourself to `AUTHORS`_.

.. _`please let us know`: https://github.com/petarmaric/fsm_strip_length_analysis/issues/new
.. _`the repository`: https://github.com/petarmaric/fsm_strip_length_analysis
.. _`AUTHORS`: https://github.com/petarmaric/fsm_strip_length_analysis/blob/master/AUTHORS
