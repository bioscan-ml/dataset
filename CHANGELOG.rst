Changelog
=========

All notable changes to bioscan_dataset will be documented here.

The format is based on `Keep a Changelog`_, and this project adheres to `Semantic Versioning`_.

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html

Categories for changes are: Added, Changed, Deprecated, Removed, Fixed, Security.


Version `1.2.0 <https://github.com/bioscan-ml/dataset/tree/v1.2.0>`__
---------------------------------------------------------------------

Release date: 2025-04-03.
`Full commit changelog <https://github.com/bioscan-ml/dataset/compare/v1.1.0...v1.2.0>`__.

This is a minor release adding some new features.
In particular, CLIBD partitioning of BIOSCAN-1M is now supported, automatic download of BIOSCAN-1M is now supported, and multiple splits can be loaded at once by joining them with ``"+"`` such as ``"pretrain+train"``.

.. _v1.2.0 Changed:

Fixed
~~~~~

-   Sped up BIOSCAN5M load times by vectorizing the image path generation process
    (`#28 <https://github.com/bioscan-ml/dataset/pull/28>`__).

-   Avoid re-download and re-extraction of splits which were already correctly present, which previously could be triggered by other splits needing to be downloaded, for example when using metasplit ``"seen"`` or ``"all"`` when some (but not all) splits were already downloaded
    (`#40 <https://github.com/bioscan-ml/dataset/pull/40>`__).

.. _v1.2.0 Added:

Added
~~~~~

-   Added support for `CLIBD <https://openreview.net/forum?id=d5HUnyByAI>`__ partitioning of BIOSCAN-1M, using argument ``partitioning_version="clibd"`` to BIOSCAN1M
    (`#25 <https://github.com/bioscan-ml/dataset/pull/25>`__,
    `#26 <https://github.com/bioscan-ml/dataset/pull/26>`__,
    `#30 <https://github.com/bioscan-ml/dataset/pull/30>`__,
    `#35 <https://github.com/bioscan-ml/dataset/pull/35>`__).

-   Added automatic download support to BIOSCAN1M.
    This includes both the metadata CSV and the image files (`#31 <https://github.com/bioscan-ml/dataset/pull/31>`__, `#37 <https://github.com/bioscan-ml/dataset/pull/37>`__), and the CLIBD partitioning data (`#33 <https://github.com/bioscan-ml/dataset/pull/33>`__).
    As with BIOSCAN5M, data is lazily downloaded, so only additional files needed for the current dataset request are downloaded.

-   Added support for combinations of splits being specified joined with ``"+"`` such as ``split="pretrain+train"``
    (`#39 <https://github.com/bioscan-ml/dataset/pull/39>`__,
    `#40 <https://github.com/bioscan-ml/dataset/pull/40>`__).

-   Added aliasing between ``"val"`` (BIOSCAN-5M) and ``"validation"`` (BIOSCAN-1M) split names
    (`#38 <https://github.com/bioscan-ml/dataset/pull/38>`__).

-   Added ``__all__`` to better support ``from bioscan_dataset import *``
    (`#41 <https://github.com/bioscan-ml/dataset/pull/41>`__).

-   Added type hinting
    (`#44 <https://github.com/bioscan-ml/dataset/pull/44>`__).

-   Added access to columns ``"processid"`` in ``BIOSCAN1M.metadata`` and both ``"area_fraction"`` and ``"scale_factor"`` in ``BIOSCAN5M.metadata``
    (`#43 <https://github.com/bioscan-ml/dataset/pull/43>`__).

-   Added more detailed ``__repr__`` information, which is shown when printing the dataset object
    (`#34 <https://github.com/bioscan-ml/dataset/pull/34>`__).

-   Improved error messages for bad split values or partitioning versions
    (`#27 <https://github.com/bioscan-ml/dataset/pull/27>`__,
    `#32 <https://github.com/bioscan-ml/dataset/pull/32>`__).

.. _v1.2.0 Documentation:

Documentation
~~~~~~~~~~~~~

-   General documentation improvements
    (`#42 <https://github.com/bioscan-ml/dataset/pull/42>`__,
    `#44 <https://github.com/bioscan-ml/dataset/pull/44>`__).


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

-   Added ``index2label`` and ``label2index`` methods to the dataset class to map between class indices and taxonomic labels
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

-   Changed color scheme to match `bioscan-browser <https://bioscan-browser.netlify.app/style-guide>`_
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
