Changelog
=========

All notable changes to bioscan_dataset will be documented here.

The format is based on `Keep a Changelog`_, and this project adheres to `Semantic Versioning`_.

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html

Categories for changes are: Added, Changed, Deprecated, Removed, Fixed, Security.


Version `1.0.1 <https://github.com/bioscan-ml/dataset/tree/v1.0.1>`__
---------------------------------------------------------------------

Release date: 2024-12-07.
`Full commit changelog <https://github.com/bioscan-ml/dataset/compare/v1.0.0...v1.0.1>`__.

This is a bugfix release to address incorrect RGB stdev values.

.. _v1.0.1 Fixed:

Fixed
~~~~~

-   RGB_STDEV for bioscan1m and bioscan5m was corrected to address a miscalculation when estimating the pixel RGB standard deviation.
    (`#2 <https://github.com/bioscan-ml/dataset/pull/2>`__)

.. _v1.0.1 Documentation:

Documentation
~~~~~~~~~~~~~

-   Corrected example import of RGB_MEAN and RGB_STDEV.
    (`#1 <https://github.com/bioscan-ml/dataset/pull/1>`__)
-   General documentation fixes and improvements.


Version `1.0.0 <https://github.com/bioscan-ml/dataset/tree/v1.0.0>`__
---------------------------------------------------------------------

Release date: 2024-12-03.
Initial release.
