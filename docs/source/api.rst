API Reference
=============

We provide :class:`~bioscan_dataset.BIOSCAN1M` and :class:`~bioscan_dataset.BIOSCAN5M` classes to load the respective `BIOSCAN-1M <BIOSCAN-1M paper_>`_ and `BIOSCAN-5M <BIOSCAN-5M paper_>`_ datasets for use within PyTorch.
These classes are subclasses of :class:`torch.utils.data.Dataset` and are designed to be used with PyTorch's :class:`~torch.utils.data.DataLoader` for batching and model training.

General usage instructions for :class:`~bioscan_dataset.BIOSCAN1M` and :class:`~bioscan_dataset.BIOSCAN5M` are provided in our :doc:`usage <index>` guide.

.. tip::
    For new projects, we recommend using :class:`~bioscan_dataset.BIOSCAN5M` instead of :class:`~bioscan_dataset.BIOSCAN1M` since the newer dataset has cleaner labels and images.
    For larger scale projects, :class:`~bioscan_dataset.BIOSCAN5M` is a superset of :class:`~bioscan_dataset.BIOSCAN1M` and will provide five times more samples to train on.
    On the other hand, if 5 million samples is too much to handle, you can ignore the ``"pretrain"`` partition (train using the ``"train"`` partition only), which reduces the dataset to less than 400k samples.

The accompanying functions :func:`~bioscan_dataset.load_bioscan1m_metadata` and :func:`~bioscan_dataset.load_bioscan5m_metadata` can be used to load the metadata from the CSV files.
This produces a :class:`~pandas.DataFrame` in the same format as is used for model training.
These functions do not need to be manually called when you are using :class:`~bioscan_dataset.BIOSCAN1M` and :class:`~bioscan_dataset.BIOSCAN5M` to work with the datasets.

.. _BIOSCAN-1M paper: https://papers.nips.cc/paper_files/paper/2023/hash/87dbbdc3a685a97ad28489a1d57c45c1-Abstract-Datasets_and_Benchmarks.html
.. _BIOSCAN-5M paper: https://arxiv.org/abs/2406.12723


BIOSCAN-1M Dataset
------------------

.. autoclass:: bioscan_dataset.BIOSCAN1M
    :members:
    :special-members: __getitem__
    :show-inheritance:

.. autofunction:: bioscan_dataset.load_bioscan1m_metadata

BIOSCAN-5M Dataset
------------------

.. autoclass:: bioscan_dataset.BIOSCAN5M
    :members:
    :special-members: __getitem__
    :show-inheritance:

.. autofunction:: bioscan_dataset.load_bioscan5m_metadata
