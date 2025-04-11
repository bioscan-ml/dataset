.. raw:: html

    <p style="margin: 0;">
        <a href="https://pypi.org/project/bioscan-dataset/"><img alt="Latest PyPI release" src="https://img.shields.io/pypi/v/bioscan-dataset.svg" style="max-width: 100%;"></a>
        <a href="https://raw.githubusercontent.com/bioscan-ml/dataset/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/pypi/l/bioscan-dataset" style="max-width: 100%;"></a>
        <a href="https://bioscan-dataset.readthedocs.io"><img alt="Documentation" src="https://img.shields.io/badge/docs-readthedocs-blue" style="max-width: 100%;"></a>
        <a href="https://github.com/psf/black"><img alt="black" src="https://img.shields.io/badge/code%20style-black-000000.svg" style="max-width: 100%;"></a>
        <a href="https://github.com/pre-commit/pre-commit"><img alt="pre-commit enabled" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&amp;logoColor=white" style="max-width: 100%;"></a>
        <a href="https://www.doi.org/10.48550/arxiv.2406.12723"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.48550/arxiv.2406.12723-blue.svg" style="max-width: 100%;"></a>
    </p>

BIOSCAN Datasets for PyTorch
============================

In this package, we provide PyTorch/torchvision style dataset classes to load the `BIOSCAN-1M <BIOSCAN-1M paper_>`_ and `BIOSCAN-5M <BIOSCAN-5M paper_>`_ datasets.

BIOSCAN-1M and 5M are large multimodal datasets for insect biodiversity monitoring, containing over 1 million and 5 million specimens, respectively.
The datasets are comprised of RGB microscopy images, `DNA barcodes <what-is-DNA-barcoding_>`_, and fine-grained, hierarchical taxonomic labels.
Every sample has both an image and a DNA barcode, but the taxonomic labels are incomplete and only extend all the way to the species level for around 9% of the specimens.
For more details about the datasets, please see the `BIOSCAN-1M paper`_ and `BIOSCAN-5M paper`_, respectively.

Documentation about this package, including the full API details, is available online at readthedocs_.


Installation
------------

The bioscan-dataset package is available on PyPI_, and the latest release can be installed into your current environment using pip_.

To install the package, run:

.. code-block:: bash

    pip install bioscan-dataset

The package source code is available on `GitHub <our repo_>`_.
If you can't wait for the next PyPI release, the latest (unstable) version can be installed with:

.. code-block:: bash

    pip install git+https://github.com/bioscan-ml/dataset.git


Usage
-----

The datasets can be used in the same way as PyTorch's `torchvision datasets <https://pytorch.org/vision/main/datasets.html#built-in-datasets>`__.
For example, to load the BIOSCAN-5M dataset:

.. code-block:: python

    from bioscan_dataset import BIOSCAN5M

    dataset = BIOSCAN5M(root="~/Datasets/bioscan/")

    for image, dna_barcode, label in dataset:
        # Do something with the image, dna_barcode, and label
        pass

To load the BIOSCAN-1M dataset:

.. code-block:: python

    from bioscan_dataset import BIOSCAN1M

    dataset = BIOSCAN1M(root="~/Datasets/bioscan/")

    for image, dna_barcode, label in dataset:
        # Do something with the image, dna_barcode, and label
        pass

Note that although BIOSCAN-5M is a superset of BIOSCAN-1M, the repeated data samples are not identical between the two due to data cleaning and processing differences.
For details, please see Appendix Q of the `BIOSCAN-5M paper`_.
Additionally, note that the splits are incompatible between the two datasets.

For these reasons, we recommend new projects use the BIOSCAN-5M dataset over BIOSCAN-1M.


Dataset download
~~~~~~~~~~~~~~~~

The dataset files can be automatically downloaded by setting the argument ``download=True`` when instantiating the dataset class:

.. code-block:: python

    dataset = BIOSCAN5M(root="~/Datasets/bioscan/", download=True)

When using the automatic download option, resources are downloaded only as needed.
The metadata is always downloaded, but the images are only downloaded if the ``"image"`` modality is selected (which it is by default, for more details see `Input modality selection`_).
Furthermore, the BIOSCAN-5M images are downloaded in a lazy manner, with splits only downloaded when they are first used.
Since 90% of the data is in the pretrain split, this means only a small fraction of the images are downloaded if this split is not used.

The BIOSCAN-1M and BIOSCAN-5M datasets both offer images in multiple versions, referred to as image packages.
The default image package is ``cropped_256``, where the images have been cropped to a bounding box around the insect, and then resized so the shorter side is 256 pixels.
Other image packages are ``cropped_full`` (cropped to a bounding box but not resized), ``original_full`` (original images at the highest resolution we provide), and ``original_256`` (uncropped images resized to 256 pixels on the shorter side).

Both `BIOSCAN1M <BS1M-class_>`_ and `BIOSCAN5M <BS5M-class_>`_ support automatically downloading the ``cropped_256`` image package, and `BIOSCAN1M <BS1M-class_>`_ additionally supports automatic download of the ``original_256`` image package.
For the other image packages, please follow the download instructions given in the `BIOSCAN-1M repository <https://github.com/bioscan-ml/BIOSCAN-1M?tab=readme-ov-file#-dataset-access>`__ and `BIOSCAN-5M repository <https://github.com/bioscan-ml/BIOSCAN-5M?tab=readme-ov-file#dataset-access>`__, respectively.
You can then set the argument ``image_package`` to work with the desired version of the images:

.. code-block:: python

    # Manually download original_full from
    # https://drive.google.com/drive/u/1/folders/1Jc57eKkeiYrnUBc9WlIp-ZS_L1bVlT-0
    # and unzip the 5 zip files into ~/Datasets/bioscan/bioscan5m/images/original_full/
    # Then load the dataset as follows:
    dataset = BIOSCAN5M(root="~/Datasets/bioscan/", image_package="original_full")


Partition/split selection
~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset class can be used to load different dataset splits.
By default, the dataset class will load the training split (``train``).

For example, to load the validation split:

.. code-block:: python

    dataset = BIOSCAN5M(root="~/Datasets/bioscan/", split="val")

In the BIOSCAN-5M dataset, the dataset is partitioned so there are ``train``, ``val``, and ``test`` splits to use for closed-world tasks (seen species), and ``key_unseen``, ``val_unseen``, and ``test_unseen`` splits to use for open-world tasks (unseen species).
These partitions only use samples labelled to species-level.

The ``pretrain`` split, which contains 90% of the data, is available for self- and semi-supervised training.
Note that these samples may include species in the unseen partition, since we don't know what species these specimens are.

Additionally, there is an ``other_heldout`` split, which contains more unseen species with either too few samples to use for testing, or a genus label which does not appear in the seen set.
This partition can be used for training a novelty detector, without exposing the detector to the species in the unseen species set.

+-------------+---------------------+-----------------------------------+-------------+------------+-----------+
| Species set | Split               | Purpose                           |  # Samples  | # Barcodes | # Species |
+=============+=====================+===================================+=============+============+===========+
| unknown     | pretrain            | self- and semi-sup. training      |   4,677,756 |  2,284,232 |         — |
+-------------+---------------------+-----------------------------------+-------------+------------+-----------+
| seen        | train               | supervision; retrieval keys       |     289,203 |    118,051 |    11,846 |
+             +---------------------+-----------------------------------+-------------+------------+-----------+
|             | val                 | model dev; retrieval queries      |      14,757 |      6,588 |     3,378 |
+             +---------------------+-----------------------------------+-------------+------------+-----------+
|             | test                | final eval; retrieval queries     |      39,373 |     18,362 |     3,483 |
+-------------+---------------------+-----------------------------------+-------------+------------+-----------+
| unseen      | key_unseen          | retrieval keys                    |      36,465 |     12,166 |       914 |
+             +---------------------+-----------------------------------+-------------+------------+-----------+
|             | val_unseen          | model dev; retrieval queries      |       8,819 |      2,442 |       903 |
+             +---------------------+-----------------------------------+-------------+------------+-----------+
|             | test_unseen         | final eval; retrieval queries     |       7,887 |      3,401 |       880 |
+-------------+---------------------+-----------------------------------+-------------+------------+-----------+
| heldout     | other_heldout       | novelty detector training         |      76,590 |     41,250 |     9,862 |
+-------------+---------------------+-----------------------------------+-------------+------------+-----------+

For more details about the BIOSCAN-5M partitioning, please see Section 4.1 of the `BIOSCAN-5M paper`_.

The dataset class also supports loading samples from multiple splits at once.
This can be done by passing a single string containing multiple split names joined with ``"+"``.
For example, to load the pretraining and training splits together:

.. code-block:: python

    dataset = BIOSCAN5M(root="~/Datasets/bioscan/", split="pretrain+train")


Input modality selection
~~~~~~~~~~~~~~~~~~~~~~~~

By default, the dataset class will load both the image and `DNA barcode <what-is-DNA-barcoding_>`_ as inputs for each sample.

This can be changed by setting the argument ``input_modality`` to either ``"image"``:

.. code-block:: python

    dataset = BIOSCAN5M(root="~/Datasets/bioscan/", modality="image")

or ``"dna"``:

.. code-block:: python

    dataset = BIOSCAN5M(root="~/Datasets/bioscan/", modality="dna")

Additionally, any column names from the metadata can be used as input modalities.
For example, to load the latitude and longitude coordinates as inputs:

.. code-block:: python

    dataset = BIOSCAN5M(root="~/Datasets/bioscan/", modality=("coord-lat", "coord-lon"))

or to load the size of the insect (in pixels) in addition to the DNA barcode:

.. code-block:: python

    dataset = BIOSCAN5M(
        root="~/Datasets/bioscan/", modality=("dna", "image_measurement_value")
    )

Multiple modalities can be selected by passing a list of column names.
Each item in the dataset will have the inputs in the same order as specified in the ``modality`` argument.

All samples have an image and a DNA barcode, but other fields may be incomplete.
Any missing values will be replaced with NaN.


Target selection
~~~~~~~~~~~~~~~~

The target label can be selected by setting the argument ``target`` to be either a taxonomic label or ``"dna_bin"``.
The `DNA BIN <what-is-DNA-BIN_>`_ is similar in granularity to (sub)species, but was generated by clustering the DNA barcodes instead of by inspecting their morphology.
The default target is ``"family"`` for  `BIOSCAN1M <BS1M-class_>`_ and ``"species"`` for `BIOSCAN5M <BS5M-class_>`_.

The target can be a single label, e.g.

.. code-block:: python

    dataset = BIOSCAN5M(root="~/Datasets/bioscan/", target_type="genus")

or a list of labels, e.g.

.. code-block:: python

    dataset = BIOSCAN5M(
        root="~/Datasets/bioscan/", target_type=["genus", "species", "dna_bin"]
    )

By default, the target values will be provided as integer indices that map to the labels for that taxonomic rank (with value ``-1`` used for missing labels), appropriate for training a classification model with cross-entropy.
This format can be controlled with the ``target_format`` argument, which takes values of either ``"index"`` or ``"text"``.
If this is set to ``target_format="text"``, the output will instead be the raw label string:

.. code-block:: python

    # Default target format is "index"
    dataset = BIOSCAN5M(
        root="~/Datasets/bioscan/", target_type="species", target_format="index"
    )
    assert dataset[0][-1] is 240

    # Using target format "text"
    dataset = BIOSCAN5M(
        root="~/Datasets/bioscan/", target_type="species", target_format="text"
    )
    assert dataset[0][-1] is "Gnamptogenys sulcata"

The default setting is ``target_format="index"``.
Note that if multiple targets types are given, each label will be returned in the same format.

To map target indices back to text labels, the dataset class provides the ``index2label`` method.
Similarly, the ``label2index`` method can be used to map text labels to indices.


Data transforms
~~~~~~~~~~~~~~~

The dataset class supports the use of data transforms for the image and DNA barcode inputs.

.. code-block:: python

    import torch
    from torchvision.transforms import v2 as transforms
    from bioscan_dataset import BIOSCAN5M
    from bioscan_dataset.bioscan5m import RGB_MEAN, RGB_STDEV

    # Create an image transform, standardizing image size and normalizing pixel values
    image_transform = transforms.Compose(
        [
            transforms.CenterCrop(256),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STDEV),
        ]
    )
    # Create a DNA transform, mapping from characters to integers and padding to a fixed length
    charmap = {"P": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
    dna_transform = lambda seq: torch.tensor(
        [charmap[char] for char in seq] + [0] * (660 - len(seq)), dtype=torch.long
    )
    # Load the dataset with the transforms applied for each sample
    ds_train = BIOSCAN5M(
        root="~/Datasets/bioscan/",
        split="train",
        transform=image_transform,
        dna_transform=dna_transform,
    )


Other resources
---------------

- Read the `BIOSCAN-1M paper`_ and `BIOSCAN-5M paper`_.
- The dataset can be explored through a web interface using our `BIOSCAN Browser`_.
- Read more about the `International Barcode of Life (iBOL) <https://ibol.org/>`__ and `BIOSCAN <https://ibol.org/bioscan/>`__ initiatives.
- See the code for the `cropping tool <https://github.com/bioscan-ml/BIOSCAN-5M/tree/main/BIOSCAN_crop_resize>`__ that was applied to the images to create the cropped image package.
- Examine the code for the `experiments <https://github.com/bioscan-ml/BIOSCAN-1M>`__ described in the BIOSCAN-1M paper.
- Examine the code for the `experiments <https://github.com/bioscan-ml/BIOSCAN-5M>`__ described in the BIOSCAN-5M paper.


Citation
--------

If you make use of the BIOSCAN-1M or BIOSCAN-5M datasets in your research, please cite the following papers as appropriate.

`BIOSCAN-5M <BIOSCAN-5M paper_>`_:

.. code-block:: bibtex

    @inproceedings{bioscan5m,
        title={{BIOSCAN-5M}: A Multimodal Dataset for Insect Biodiversity},
        booktitle={Advances in Neural Information Processing Systems},
        author={Zahra Gharaee and Scott C. Lowe and ZeMing Gong and Pablo Millan Arias
            and Nicholas Pellegrino and Austin T. Wang and Joakim Bruslund Haurum
            and Iuliia Zarubiieva and Lila Kari and Dirk Steinke and Graham W. Taylor
            and Paul Fieguth and Angel X. Chang
        },
        editor={A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
        pages={36285--36313},
        publisher={Curran Associates, Inc.},
        year={2024},
        volume={37},
        url={https://proceedings.neurips.cc/paper_files/paper/2024/file/3fdbb472813041c9ecef04c20c2b1e5a-Paper-Datasets_and_Benchmarks_Track.pdf},
    }

`BIOSCAN-1M <BIOSCAN-1M paper_>`_:

.. code-block:: bibtex

    @inproceedings{bioscan1m,
        title={A Step Towards Worldwide Biodiversity Assessment: The {BIOSCAN-1M} Insect Dataset},
        booktitle={Advances in Neural Information Processing Systems},
        author={Gharaee, Z. and Gong, Z. and Pellegrino, N. and Zarubiieva, I.
            and Haurum, J. B. and Lowe, S. C. and McKeown, J. T. A. and Ho, C. Y.
            and McLeod, J. and Wei, Y. C. and Agda, J. and Ratnasingham, S.
            and Steinke, D. and Chang, A. X. and Taylor, G. W. and Fieguth, P.
        },
        editor={A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
        pages={43593--43619},
        publisher={Curran Associates, Inc.},
        year={2023},
        volume={36},
        url={https://proceedings.neurips.cc/paper_files/paper/2023/file/87dbbdc3a685a97ad28489a1d57c45c1-Paper-Datasets_and_Benchmarks.pdf},
    }

If you use the CLIBD partitioning scheme for BIOSCAN-1M, please also consider citing the `CLIBD paper <https://arxiv.org/abs/2405.17537>`_.

.. code-block:: bibtex

    @inproceedings{clibd,
        title={{CLIBD}: Bridging Vision and Genomics for Biodiversity Monitoring at Scale},
        author={ZeMing Gong and Austin Wang and Xiaoliang Huo and Joakim Bruslund Haurum
            and Scott C. Lowe and Graham W. Taylor and Angel X Chang
        },
        booktitle={The Thirteenth International Conference on Learning Representations},
        year={2025},
        url={https://openreview.net/forum?id=d5HUnyByAI},
    }

.. _BIOSCAN Browser: https://bioscan-browser.netlify.app/
.. _BIOSCAN-1M paper: https://papers.nips.cc/paper_files/paper/2023/hash/87dbbdc3a685a97ad28489a1d57c45c1-Abstract-Datasets_and_Benchmarks.html
.. _BIOSCAN-5M paper: https://arxiv.org/abs/2406.12723
.. _BS1M-class: https://bioscan-dataset.readthedocs.io/en/stable/api.html#bioscan_dataset.BIOSCAN1M
.. _BS5M-class: https://bioscan-dataset.readthedocs.io/en/stable/api.html#bioscan_dataset.BIOSCAN5M
.. _our repo: https://github.com/bioscan-ml/dataset
.. _pip: https://pip.pypa.io/
.. _PyPI: https://pypi.org/project/bioscan-dataset/
.. _readthedocs: https://bioscan-dataset.readthedocs.io
.. _what-is-DNA-barcoding: https://www.ibol.org/phase1/about-us/what-is-dna-barcoding/
.. _what-is-DNA-BIN: https://portal.boldsystems.org/bin
