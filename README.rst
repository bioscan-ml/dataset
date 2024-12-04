|pre-commit| |black|

BIOSCAN Datasets for PyTorch
============================

In this package, we provide PyTorch/torchvision style dataset classes to load the `BIOSCAN-1M <BS1M-paper_>`_ and `BIOSCAN-5M <BS5M-paper_>`_ datasets.

BIOSCAN-1M and 5M are large multimodal datasets for insect biodiversity monitoring, containing over 1 million and 5 million specimens, respectively.
The datasets are comprised of RGB microscopy images, DNA barcodes, and fine-grained, hierarchical taxonomic labels.
Every sample has both an image and a DNA barcode, but the taxonomic labels are incomplete and only extend all the way to the species level for around 9% of the specimens.


Installation
------------

To install the package, run:

.. code-block:: bash

   pip install bioscan-dataset


Usage
-----

The datasets can be used in the same way as PyTorch's torchvision datasets.
For example, to load the BIOSCAN-1M dataset:

.. code-block:: python

   from bioscan_dataset import BIOSCAN1M

   dataset = BIOSCAN1M(root="~/Datasets/bioscan/bioscan-1m/")

   for (image, dna_barcode), label in dataset:
       # Do something with the image, dna_barcode, and label
       pass

To load the BIOSCAN-5M dataset:

.. code-block:: python

   from bioscan_dataset import BIOSCAN5M

   dataset = BIOSCAN5M(root="~/Datasets/bioscan/bioscan-5m/")

   for (image, dna_barcode), label in dataset:
       # Do something with the image, dna_barcode, and label
       pass


Note that although BIOSCAN-5M is a superset of BIOSCAN-1M, the repeated data samples are not identical between the two due to data cleaning and processing differences.
Additionally, note that the splits are incompatible between the two datasets.
For details, see the `BIOSCAN-5M paper <BS5M-paper_>`_.

For these reasons, we recommend new projects use the BIOSCAN-5M dataset over BIOSCAN-1M.


Dataset download
~~~~~~~~~~~~~~~~

For BIOSCAN-5M, the dataset class supports automatically downloading the ``cropped_256`` image package (which is the default package).
This can be performed by setting the argument ``download=True``:

.. code-block:: python

   dataset = BIOSCAN5M(root="~/Datasets/bioscan/bioscan-5m/", download=True)

To use a different image package, follow the download instructions given in the `BIOSCAN-5M repository <https://github.com/bioscan-ml/BIOSCAN-5M?tab=readme-ov-file#dataset-access>`_, then set the argument ``image_package`` to the desired package name, e.g.

.. code-block:: python

   # Manually download original_full from
   # https://drive.google.com/drive/u/1/folders/1Jc57eKkeiYrnUBc9WlIp-ZS_L1bVlT-0
   # and unzip the 5 zip files into ~/Datasets/bioscan/bioscan-5m/original_full/
   # Then load the dataset as follows:
   dataset = BIOSCAN5M(
       root="~/Datasets/bioscan/bioscan-5m/", image_package="original_full"
   )

For BIOSCAN-1M, automatic dataset download is not supported and so the dataset must be manually downloaded.
See the `BIOSCAN-1M repository <https://github.com/bioscan-ml/BIOSCAN-1M?tab=readme-ov-file#-dataset-access>`_ for download instructions.


Partition/split selection
~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset class can be used to load different dataset splits.
By default, the dataset class will load the training split (``train``).

For example, to load the validation split:

.. code-block:: python

   dataset = BIOSCAN5M(root="~/Datasets/bioscan/bioscan-5m/", split="val")

In the BIOSCAN-5M dataset, the dataset is partitioned so there are ``train``, ``val``, and ``test`` splits to use for closed-world tasks (seen species), and ``key_unseen``, ``val_unseen``, and ``test_unseen`` splits to use for open-world tasks (unseen species).
These partitions only use samples labelled to species-level.

The ``pretrain`` split, which contains 90% of the data, is available for self- and semi-supervised training.
Note that these samples may include species in the unseen partition, since we don't know what species these specimens are.

Additionally, there is an ``other_heldout`` split, which contains more unseen species with either too samples to use for testing, or a genus label which does not appear in the seen set.
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

For more details about the BIOSCAN-5M partitioning, please see the `BIOSCAN-5M paper <BS5M-paper_>`_.


Input modality selection
~~~~~~~~~~~~~~~~~~~~~~~~

By default, the dataset class will load both the image and DNA barcode as inputs for each sample.

This can be changed by setting the argument ``input_modality`` to either ``"image"``:

.. code-block:: python

   dataset = BIOSCAN5M(root="~/Datasets/bioscan/bioscan-5m/", modality="image")

or ``"dna"``:

.. code-block:: python

   dataset = BIOSCAN5M(root="~/Datasets/bioscan/bioscan-5m/", modality="dna")


Target selection
~~~~~~~~~~~~~~~~

The target label can be selected by setting the argument ``target`` to be either a taxonomic label or ``dna_bin``.
The DNA BIN is similar in granularity to subspecies, but was generated by clustering the DNA barcodes instead of morphology.
The default target is ``"family"`` for BIOSCAN1M and ``"species"`` for BIOSCAN5M.

The target can be a single label, e.g.

.. code-block:: python

   dataset = BIOSCAN5M(root="~/Datasets/bioscan/bioscan-5m/", target_type="genus")

or a list of labels, e.g.

.. code-block:: python

   dataset = BIOSCAN5M(
       root="~/Datasets/bioscan/bioscan-5m/", target_type=["genus", "species", "dna_bin"]
   )

The value of the target yielded for a data sample is an integer corresponding to the index of its label.


Data transforms
~~~~~~~~~~~~~~~

The dataset class supports the use of data transforms for the image and DNA barcode inputs.

.. code-block:: python

   import torch
   import torchvision.transforms as transforms
   from bioscan_dataset import BIOSCAN5M
   from bioscan_dataset.BIOSCAN5M import RGB_MEAN, RGB_STDEV

   # Create an image transform, standardizing image size and normalizing pixel values
   image_transform = transforms.Compose(
       [
           transforms.Resize(224),
           transforms.ToTensor(),
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
       root="~/Datasets/bioscan/bioscan-5m/",
       split="train",
       transform=image_transform,
       dna_transform=dna_transform,
   )


Size and geolocation metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The BIOSCAN-5M dataset also contains insect size and geolocation metadata.
Loading this metadata is not yet supported by the BIOSCAN5M pytorch dataset class.
In the meantime, users of the dataset are welcome to explore this metadata themselves.


Other resources
---------------

- Read the `BIOSCAN-1M paper <BS1M-paper_>`_ and `BIOSCAN-5M paper <BS5M-paper_>`_.
- The dataset can be explored through a web interface using our `BIOSCAN Browser <https://bioscan-browser.netlify.app/>`_.
- Read more about the `International Barcode of Life (iBOL) <https://ibol.org/>`_ and `BIOSCAN <https://ibol.org/bioscan/>`_ initiatives.
- See the code for the `cropping tool <https://github.com/bioscan-ml/BIOSCAN-5M/tree/main/BIOSCAN_crop_resize>`_ that was applied to the images to create the cropped image package.
- Examine the code for the `experiments <https://github.com/bioscan-ml/BIOSCAN-5M>`_ described in the BIOSCAN-5M paper.


Citation
--------

If you make use of the BIOSCAN-1M or BIOSCAN-5M datasets in your research, please cite the following papers as appropriate.

`BIOSCAN-5M <BS5M-paper_>`_:

.. code-block:: bibtex

   @misc{bioscan5m,
      title={{BIOSCAN-5M}: A Multimodal Dataset for Insect Biodiversity},
      author={Zahra Gharaee and Scott C. Lowe and ZeMing Gong and Pablo Millan Arias
         and Nicholas Pellegrino and Austin T. Wang and Joakim Bruslund Haurum
         and Iuliia Zarubiieva and Lila Kari and Dirk Steinke and Graham W. Taylor
         and Paul Fieguth and Angel X. Chang
      },
      year={2024},
      eprint={2406.12723},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      doi={10.48550/arxiv.2406.12723},
   }

`BIOSCAN-1M <BS1M-paper_>`_:

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

.. _BS1M-paper: https://papers.nips.cc/paper_files/paper/2023/hash/87dbbdc3a685a97ad28489a1d57c45c1-Abstract-Datasets_and_Benchmarks.html
.. _BS5M-paper: https://arxiv.org/abs/2406.12723

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: black
