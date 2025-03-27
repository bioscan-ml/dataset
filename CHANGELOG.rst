Changelog
=========

All notable changes to bioscan_dataset will be documented here.

The format is based on `Keep a Changelog`_, and this project adheres to `Semantic Versioning`_.

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html

Categories for changes are: Added, Changed, Deprecated, Removed, Fixed, Security.


Version `1.1.0 <https://github.com/bioscan-ml/dataset/tree/v1.1.0>`__
---------------------------------------------------------------------

Release date: 2025-03-27.
`Full commit changelog <https://github.com/bioscan-ml/dataset/compare/v1.0.1...v1.1.0>`__.

This is a minor release adding some new features.

.. _v1.1.0 Added:

Added
~~~~~

-   Added ``target_format`` argument which controls whether taxonomic labels are returned by ``__getitem__`` as a strings or integers indicating the class index
    (`#10 <https://github.com/bioscan-ml/dataset/pull/10>`__).
    Thanks to `@xl-huo <https://github.com/xl-huo>`_ for contributing this.

-   Added ``index2label`` and ``label2index`` properties to the dataset class to map between class indices and taxonomic labels
    (`#12 <https://github.com/bioscan-ml/dataset/pull/12>`__,
    `#23 <https://github.com/bioscan-ml/dataset/pull/23>`__).

-   Added support for arbitrary modality names, which are taken from the metadata, without the option to apply a transform to the data
    (`#13 <https://github.com/bioscan-ml/dataset/pull/13>`__).

-   Added ``image_package`` argument to BIOSCAN1M, to select the image package to use, as was alreaday implemented for BIOSCAN5M
    (`#15 <https://github.com/bioscan-ml/dataset/pull/15>`__).

-   Added an warning to BIOSCAN1M that is automatically raised if one of the requested target ranks is incompatible with the selected ``partitioning_version``
    (`#18 <https://github.com/bioscan-ml/dataset/pull/18>`__).
    Thanks `@kevinkasa <https://github.com/kevinkasa>`__ for highlighting this.

.. _v1.1.0 Documentation:

Documentation
~~~~~~~~~~~~~

-   Changed color scheme to match `bioscan-browser <https://bioscan-browser.netlify.app/style-guide_>`_
    (`#4 <https://github.com/bioscan-ml/dataset/pull/4>`__).
    Thanks to `@annavik <https://github.com/annavik>`_ for contributing to this.

-   Corrected example usage to use a single tuple, not nested
    (`#5 <https://github.com/bioscan-ml/dataset/pull/5>`__).
    Thanks to `@xl-huo <https://github.com/xl-huo>`_ for reporting this.

-   General documentation improvements
    (`#3 <https://github.com/bioscan-ml/dataset/pull/3>`__,
    `#11 <https://github.com/bioscan-ml/dataset/pull/11>`__,
    `#14 <https://github.com/bioscan-ml/dataset/pull/14>`__,
    `#16 <https://github.com/bioscan-ml/dataset/pull/16>`__,
    `#17 <https://github.com/bioscan-ml/dataset/pull/17>`__,
    `#22 <https://github.com/bioscan-ml/dataset/pull/22>`__).


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
