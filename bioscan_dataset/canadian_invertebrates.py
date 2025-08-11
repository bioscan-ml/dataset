r"""
Canadian Invertebrates 1.5M PyTorch Dataset.

:Date: 2025-05-02
:Authors:
    - Vivian Phung <vivian.phung17@gmail.com>
    - Scott C. Lowe <scott.code.lowe@gmail.com>
:Copyright: 2025, Scott C. Lowe and Vivian Phung
:License: MIT
"""

import os
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_url

__all__ = ["CanadianInvertebrates", "load_canadian_invertebrates_metadata"]

COLUMN_DTYPES = {
    "processid": str,
    "sampleid": str,
    "dna_bin": "category",
    "phylum": "category",
    "class": "category",
    "order": "category",
    "family": "category",
    "genus": "category",
    "species": "category",
    "dna_barcode": str,
    "sequence_len": float,
    "split": "category",
}

USECOLS = [
    "processid",
    "dna_bin",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "dna_barcode",
    "split",
]

COLUMN_ALIASES = {"bin_uri": "dna_bin", "nucleotides": "dna_barcode"}
VALID_SPLITS = ["pretrain", "train", "val", "test", "test_unseen"]
SPLIT_ALIASES = {"validation": "val"}
VALID_METASPLITS = ["all", "seen", "unseen"]
SEEN_SPLITS = ["train", "val", "test"]
UNSEEN_SPLITS = ["test_unseen"]


def explode_metasplit(metasplit: str, verify: bool = False) -> Set[str]:
    r"""
    Convert a metasplit string into its set of constituent splits.

    Parameters
    ----------
    metasplit : str
        The metasplit to explode.
    verify : bool, default=False
        If ``True``, verify that the constitutent splits are valid.

    Returns
    -------
    set of str
        The canonical splits within the metasplit.

    Examples
    --------
    >>> explode_metasplit("pretrain+train")
    {'pretrain', 'train'}
    >>> explode_metasplit("seen")
    {'train', 'val', 'test'}
    >>> explode_metasplit("train")
    {'train'}
    >>> explode_metasplit("validation")
    {'val'}
    """
    if metasplit is None:
        metasplit = "all"
    split_list = [s.strip() for s in metasplit.split("+")]
    split_list = [SPLIT_ALIASES.get(s, s) for s in split_list]
    split_set = set(split_list)
    if "all" in split_list:
        split_set.remove("all")
        split_set |= set(VALID_SPLITS)
    if "seen" in split_list:
        split_set.remove("seen")
        split_set |= set(SEEN_SPLITS)
    if "unseen" in split_list:
        split_set.remove("unseen")
        split_set |= set(UNSEEN_SPLITS)

    if verify:
        # Verify the constituent splits are valid
        invalid_splits = split_set - set(VALID_SPLITS)
        if invalid_splits:
            msg_valid_names = f"Valid split names are: {', '.join(repr(s) for s in VALID_METASPLITS + VALID_SPLITS)}."
            if split_set == {metasplit}:
                raise ValueError(f"Invalid split name {repr(metasplit)}. {msg_valid_names}")
            plural = "s" if len(invalid_splits) > 1 else ""
            raise ValueError(
                f"Invalid split name{plural} {', '.join(repr(s) for s in invalid_splits)} within requested metasplit"
                f" {repr(metasplit)}. {msg_valid_names}"
            )

    return split_set


class MetadataDtype(Enum):
    DEFAULT = "CanadianInvertebrates_default_dtypes"


def load_canadian_invertebrates_metadata(
    metadata_path,
    max_nucleotides: Union[int, None] = 660,
    reduce_repeated_barcodes: bool = False,
    split: Optional[str] = None,
    dtype: Union[str, dict, None] = MetadataDtype.DEFAULT,
    **kwargs,
) -> pandas.DataFrame:
    r"""
    Load Canadian Invertebrates dataset metadata from its CSV file and prepare it for training.

    .. versionadded:: 1.4.0

    Parameters
    ----------
    metadata_path : str
        Path to the metadata CSV file.

    max_nucleotides : int, default=660
        Maximum nucleotide sequence length to keep for the DNA barcodes.
        Set to ``None`` to keep the original data without truncation.

        .. note::
            COI DNA barcodes are typically 658 base pairs long for insects
            (`Elbrecht et al., 2019 <https://doi.org/10.7717/peerj.7745>`_),
            and an additional two base pairs are included as a buffer for the
            primer sequence.
            Although the Canadian Invertebrates dataset itself contains longer sequences,
            characters after the first 660 base pairs are likely to be inaccurate
            reads, and not part of the DNA barcode.
            Hence we recommend limiting the DNA barcode to the first 660 nucleotides.
            If you don't know much about DNA barcodes, you probably shouldn't
            change this parameter.

    reduce_repeated_barcodes : bool, default=False
        Whether to reduce the dataset to only one sample per barcode.
        If ``True``, duplicated barcodes are removed after truncating the barcodes to
        the length specified by ``max_nucleotides`` and stripping trailing Ns.
        If ``False`` (default) no reduction is performed.

        .. note::
            This version of the dataset is already reduced to one sample per full-length
            barcode compared to the original Canadian Invertebrates 1.5M dataset
            (`deWaard et al., 2019 <https://zenodo.org/records/3497622>`_).
            Even with the parameter ``reduce_repeated_barcodes=False``, the dataset
            will have at most 965,289 samples, and not 1.5M.
            For more details on the dataset preprocessing steps, see
            `Millan Arias et al. (2024) <https://arxiv.org/abs/2311.02401>`_.

    split : str, optional
        The dataset partition to return.
        One of:

        - ``"pretrain"``
        - ``"train"``
        - ``"val"``
        - ``"test"``
        - ``"test_unseen"``
        - ``"all"``, which is the union of all splits
        - ``"seen"``, which is the union of {train, val, test}
        - ``"unseen"``, which is an alias for ``"test_unseen"``

        If ``split`` is ``None`` or ``"all"`` (default), the data is not filtered by
        partition and the dataframe will contain every sample in the dataset.

        The ``split`` parameter can also be specified as collection of partitions
        joined by ``"+"``. For example, ``"pretrain+train"`` will filter the metadata
        to samples in either the pretraining or training partitions.

    **kwargs
        Additional keyword arguments to pass to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        The metadata DataFrame.
    """
    if dtype == MetadataDtype.DEFAULT:
        # Use our default column data types
        dtype = COLUMN_DTYPES
    # Read the metadata CSV file
    df = pandas.read_csv(metadata_path, dtype=dtype, **kwargs)
    # Truncate the DNA barcodes to the specified length
    if max_nucleotides is not None:
        df["dna_barcode"] = df["dna_barcode"].str[:max_nucleotides]
    # Reduce the dataset to only one sample per barcode
    if reduce_repeated_barcodes:
        # Shuffle the data order, to avoid bias in the subsampling that could be induced
        # by the order in which the data was collected.
        df = df.sample(frac=1, random_state=0)
        # Drop duplicated barcodes
        df["dna_barcode_strip"] = df["dna_barcode"].str.rstrip("N")
        df = df.drop_duplicates(subset=["dna_barcode_strip"])
        df.drop(columns=["dna_barcode_strip"], inplace=True)
        # Re-order the data (reverting the shuffle)
        df = df.sort_index()
    # Filter to just the split of interest
    if split is not None and split != "all":
        # Split the string by "+" to handle custom metasplits
        split_set = explode_metasplit(split, verify=True)
        # Filter the DataFrame to just the requested splits
        df = df.loc[df["split"].isin(split_set)]
    # Add index columns to use for targets
    label_cols = [
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "dna_bin",
    ]
    for c in label_cols:
        df[c] = df[c].astype("category")
        df[c + "_index"] = df[c].cat.codes
    return df


load_metadata = load_canadian_invertebrates_metadata


class CanadianInvertebrates(Dataset):
    r"""
    `Canadian Invertebrates <https://huggingface.co/datasets/bioscan-ml/CanadianInvertebrates-ML>`_ Dataset.

    .. versionadded:: 1.4.0

    Parameters
    ----------
    root : str
        The root directory, to contain the downloaded tarball files and
        CanadianInvertebrates data directory.

    split : str, default="train"
        The dataset partition.
        One of:

        - ``"pretrain"``
        - ``"train"``
        - ``"val"``
        - ``"test"``
        - ``"test_unseen"``
        - ``"all"``, which is the union of all splits
        - ``"seen"``, which is the union of {train, val, test}
        - ``"unseen"``, which is an alias for ``"test_unseen"``

        Set to ``"all"`` to include all splits.

        The ``split`` parameter can also be specified as collection of partitions
        joined by ``"+"``. For example, ``split="pretrain+train"`` will return a dataset
        comprised of the pretraining and training partitions.

    modality : str or Iterable[str], default=("dna")
        Which data modalities to use. This dataset only has one modality: the
        DNA barcode. The ``modality`` parameter is only present to provide a consistent
        interface with the other dataset classes in this package.

    reduce_repeated_barcodes : bool, default=False
        Whether to reduce the dataset to only one sample per barcode.
        If ``True``, duplicated barcodes are removed after truncating the barcodes to
        the length specified by ``max_nucleotides`` and stripping trailing Ns.
        If ``False`` (default) no additional reduction of repeated barcodes is
        performed.

        .. note::
            This version of the dataset is already reduced to one sample per full-length
            barcode compared to the original Canadian Invertebrates 1.5M dataset
            (`deWaard et al., 2019 <https://zenodo.org/records/3497622>`_).
            Even with the parameter ``reduce_repeated_barcodes=False``, the dataset
            will have at most 965,289 samples, and not 1.5M.
            For more details on the dataset preprocessing steps, see
            `Millan Arias et al. (2024) <https://arxiv.org/abs/2311.02401>`_.

    max_nucleotides : int, default=660
        Maximum number of nucleotides to keep in the DNA barcode.
        Set to ``None`` to keep the original data without truncation.

        .. note::
            COI DNA barcodes are typically 658 base pairs long for insects
            (`Elbrecht et al., 2019 <https://doi.org/10.7717/peerj.7745>`_),
            and an additional two base pairs are included as a buffer for the
            primer sequence.
            Hence we recommend limiting the DNA barcode to the first 660 nucleotides.
            If you don't know much about DNA barcodes, you probably shouldn't
            change this parameter.

    target_type : str or Iterable[str], default="species"
        Type of target to use. One of, or a list of:

        - ``"phylum"``
        - ``"class"``
        - ``"order"``
        - ``"family"``
        - ``"subfamily"``
        - ``"genus"``
        - ``"species"``
        - ``"dna_bin"`` (a species-level label derived from
          `DNA barcode clustering by BOLD <https://portal.boldsystems.org/bin>`_)

    target_format : str, default="index"
        Format in which the targets will be returned. One of:
        ``"index"``, ``"text"``.
        If this is set to ``"index"`` (default), target(s) will each be returned as
        integer indices, each of which corresponds to a value for that taxonomic rank in
        a look-up-table.
        Missing values will be filled with ``-1``.
        This format is appropriate for use in classification tasks.
        If this is set to ``"text"``, the target(s) will each be returned as a string,
        appropriate for processing with language models.

    output_format : str, default="tuple"
        Format in which :meth:`__getitem__` will be returned. One of:
        ``"tuple"``, ``"dict"``.
        If this is set to ``"tuple"`` (default), all modalities and targets will be
        returned together as a single tuple.
        If this is set to ``"dict"``, the output will be returned as a dictionary
        containing the modalities and targets as separate keys.

    dna_transform : Callable, optional
        DNA barcode transformation pipeline.

    target_transform : Callable, optional
        Label transformation pipeline.

    download : bool, default=False
        If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.

    Attributes
    ----------
    metadata : pandas.DataFrame
        The metadata associated with the samples in the select split, loaded using
        :func:`load_canadian_invertebrates_metadata`.
    """

    base_folder = "CanadianInvertebrates"
    meta = {
        "urls": [
            "https://huggingface.co/datasets/bioscan-ml/CanadianInvertebrates-ML/resolve/a8cb2b5/CanInv_metadata.csv",
        ],
        "filename": "CanInv_metadata.csv",
        "csv_md5": "b93517975176a12f3c07f21be8c37fa5",
    }

    def __init__(
        self,
        root,
        split: str = "train",
        modality: Union[str, Iterable[str]] = ("dna"),
        reduce_repeated_barcodes: bool = False,
        max_nucleotides: Union[int, None] = 660,
        target_type: Union[str, Iterable[str]] = "species",
        target_format: str = "index",
        output_format: str = "tuple",
        dna_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        root = os.path.expanduser(root)

        self.metadata = None
        self.root = root
        self.metadata_path = os.path.join(self.root, self.base_folder, self.meta["filename"])

        self.split = SPLIT_ALIASES.get(split, split)
        self.target_format = target_format
        self.output_format = "dict" if output_format == "dictionary" else output_format
        self.reduce_repeated_barcodes = reduce_repeated_barcodes
        self.max_nucleotides = max_nucleotides
        self.dna_transform = dna_transform
        self.target_transform = target_transform

        if isinstance(modality, str):
            self.modality = [modality]
        else:
            self.modality = list(modality)

        if isinstance(target_type, str):
            self.target_type = [target_type]
        else:
            self.target_type = list(target_type)
        self.target_type = [COLUMN_ALIASES.get(t, t) for t in self.target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if self.target_format not in ["index", "text"]:
            raise ValueError(f"Unknown target_format: {repr(self.target_format)}")

        if download:
            self.download()

        if not self._check_integrity():
            raise EnvironmentError(f"{type(self).__name__} dataset not found, incomplete, or corrupted: {self.root}.")

        self._load_metadata()

    def index2label(
        self,
        index: Union[int, List[int], npt.NDArray[np.int_]],
        column: Optional[str] = None,
    ) -> Union[str, npt.NDArray[np.str_]]:
        r"""
        Convert target's integer index to text label.

        Parameters
        ----------
        index : int or array_like[int]
            The integer index or indices to map to labels.
        column : str, default=same as ``self.target_type``
            The dataset column name to map.
            This should be one of the possible values for ``target_type``.
            By default, the column name is the ``target_type`` used for the class,
            provided it is a single value.

        Returns
        -------
        str or numpy.array[str]
            The text label or labels corresponding to the integer index or indices
            in the specified column.
            Entries containing missing values, indicated by negative indices, are mapped
            to an empty string.

        Examples
        --------
        >>> dataset.index2label(29, "order")
        'Diptera'
        >>> dataset.index2label([4, 9, -1, 4], "order")
        array(['Anomopoda', 'Araneae', '', 'Anomopoda'], dtype=object)
        """
        if column is not None:
            pass
        elif len(self.target_type) == 1:
            column = self.target_type[0]
        else:
            raise ValueError("column must be specified if there isn't a single target_type")
        if not hasattr(index, "__len__"):
            # Single index
            if index < 0:
                return ""
            return self.metadata[column].cat.categories[index]
        if isinstance(index, str):
            raise TypeError(
                f"index must be an int or array-like of ints, not a string: {repr(index)}."
                " Did you mean to call label2index?"
            )
        index = np.asarray(index)
        out = self.metadata[column].cat.categories[index]
        out = np.asarray(out)
        out[index < 0] = ""
        return out

    def label2index(
        self,
        label: Union[str, Iterable[str]],
        column: Optional[str] = None,
    ) -> Union[int, npt.NDArray[np.int_]]:
        r"""
        Convert target's text label to integer index.

        Parameters
        ----------
        label : str or Iterable[str]
            The text label or labels to map to integer indices.
        column : str, default=same as ``self.target_type``
            The dataset column name to map.
            This should be one of the possible values for ``target_type``.
            By default, the column name is the ``target_type`` used for the class,
            provided it is a single value.

        Returns
        -------
        int or numpy.array[int]
            The integer index or indices corresponding to the text label or labels
            in the specified column.
            Entries containing missing values, indicated by empty strings or NaN values,
            are mapped to ``-1``.

        Examples
        --------
        >>> dataset.label2index("Diptera", "order")
        29
        >>> dataset.label2index(["Diptera", "Lepidoptera", "", "Diptera"], "order")
        array([29, 45, -1, 29])
        """
        if column is not None:
            pass
        elif len(self.target_type) == 1:
            column = self.target_type[0]
        else:
            raise ValueError("column must be specified if there isn't a single target_type")
        if pandas.isna(label) or label == "":
            # Single index
            return -1
        if isinstance(label, str):
            try:
                return self.metadata[column].cat.categories.get_loc(label)
            except KeyError:
                raise KeyError(f"Label {repr(label)} not found in metadata column {repr(column)}") from None
        if isinstance(label, (int, np.integer)):
            raise TypeError(
                f"label must be a string or list of strings, not an int: {repr(label)}."
                " Did you mean to call index2label?"
            )
        labels = label
        try:
            out = [
                -1 if lab == "" or pandas.isna(lab) else self.metadata[column].cat.categories.get_loc(lab)
                for lab in labels
            ]
        except KeyError:
            raise KeyError(f"Label {repr(label)} not found in metadata column {repr(column)}") from None
        out = np.asarray(out)
        return out

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        r"""
        Get a sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple or dict
            If ``output_format="tuple"``, the output will be a tuple containing:

            - dna : str or Any
                The DNA barcode, if the ``"dna"`` modality is requested, optionally
                transformed by the ``dna_transform`` pipeline.
            - \*modalities : Any
                Any other modalities requested, as specified in the ``modality`` parameter.
                The data is extracted from the appropriate column in the metadata csv file,
                without any transformations. Missing values will be filled with NaN.
            - target : int or Tuple[int, ...] or str or Tuple[str, ...] or None
                The target(s), optionally transformed by the ``target_transform`` pipeline.
                If ``target_format="index"``, the target(s) will be returned as integer
                indices, with missing values filled with ``-1``.
                If ``target_format="text"``, the target(s) will be returned as a string.
                If there are multiple targets, they will be returned as a tuple.
                If ``target_type`` is an empty list, the output ``target`` will be ``None``.

            If ``output_format="dict"``, the output will be a dictionary with keys
            and values as follows:

            - keys for each of the modalities specified in the ``modality`` parameter,
              with corresponding values as described above.
            - keys for each of the targets specified in ``target_type``,
              with corresponding value equal to that target's label
              (e.g. ``out["species"] == "Gnamptogenys sulcata"``)
            - for each of the keys in ``target_type``, the corresponding index column (``{target}_index``),
              with value equal to that target's index
              (e.g. ``out["species_index"] == 240``)
            - the key ``"target"``, whose contents are as described above
        """
        sample = self.metadata.iloc[index]
        values = []
        for modality in self.modality:
            if modality in ["dna_barcode", "dna", "barcode"]:
                X = sample["dna_barcode"]
                if self.dna_transform is not None:
                    X = self.dna_transform(X)
            elif modality in self.metadata.columns:
                X = sample[modality]
            else:
                raise ValueError(f"Unfamiliar modality: {repr(modality)}")
            values.append((modality, X))

        target = []
        for t in self.target_type:
            if self.target_format == "index":
                target.append(sample[f"{t}_index"])
            elif self.target_format == "text":
                target.append(sample[t])
            else:
                raise ValueError(f"Unknown target_format: {repr(self.target_format)}")
            if self.output_format == "dict":
                values.append((t, sample[t]))
                key = f"{t}_index"
                if key in sample:
                    values.append((key, sample[key]))

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None
        values.append(("target", target))

        if self.output_format == "tuple":
            return tuple(v for _, v in values)
        elif self.output_format == "dict":
            return dict(values)
        else:
            raise ValueError(f"Unknown output_format: {repr(self.output_format)}")

    def _check_integrity_metadata(self, verbose=1) -> bool:
        p = self.metadata_path
        check = check_integrity(p, self.meta["csv_md5"])
        if verbose >= 1 and not check:
            if not os.path.exists(p):
                print(f"File missing: {p}")
            else:
                print(f"File invalid: {p}")
        if verbose >= 2 and check:
            print(f"File present: {p}")
        return check

    def _check_integrity(self, verbose=1) -> bool:
        r"""
        Check if the dataset is already downloaded and extracted.

        Parameters
        ----------
        verbose : int, default=1
            Verbosity level.

        Returns
        -------
        bool
            True if the dataset is already downloaded and extracted, False otherwise.
        """
        check = True
        check &= self._check_integrity_metadata(verbose=verbose)
        if not check and verbose >= 1:
            print(f"{type(self).__name__} dataset not found, incomplete, or corrupted.")
        return check

    def _download_metadata(self, verbose=1) -> None:
        if self._check_integrity_metadata(verbose=verbose):
            if verbose >= 1:
                print("Metadata CSV file already downloaded and verified")
            return
        target_dir = os.path.join(self.root, self.base_folder)
        os.makedirs(target_dir, exist_ok=True)
        download_url(self.meta["urls"][0], target_dir, md5=self.meta["csv_md5"])

    def download(self) -> None:
        r"""
        Download and extract the data.
        """
        self._download_metadata()

    def _load_metadata(self) -> pandas.DataFrame:
        r"""
        Load metadata from CSV file and prepare it for training.
        """
        self.metadata = load_metadata(
            self.metadata_path,
            max_nucleotides=self.max_nucleotides,
            reduce_repeated_barcodes=self.reduce_repeated_barcodes,
            split=self.split,
            usecols=USECOLS,
        )
        return self.metadata

    def extra_repr(self) -> str:
        xr = ""
        xr += f"split: {repr(self.split)}\n"
        xr += f"modality: {repr(self.modality)}\n"
        if self.reduce_repeated_barcodes:
            xr += f"reduce_repeated_barcodes: {repr(self.reduce_repeated_barcodes)}\n"
        has_dna_modality = any(m in self.modality for m in ["dna", "dna_barcode", "barcode"])
        if has_dna_modality and self.max_nucleotides != 660:
            xr += f"max_nucleotides: {repr(self.max_nucleotides)}\n"
        xr += f"target_type: {repr(self.target_type)}\n"
        if len(self.target_type) > 0:
            xr += f"target_format: {repr(self.target_format)}\n"
        xr += f"output_format: {repr(self.output_format)}"
        if has_dna_modality and self.dna_transform is not None:
            xr += f"\ndna_transform: {repr(self.dna_transform)}"
        return xr

    def __repr__(self) -> str:
        x = f"{self.__class__.__name__} Dataset\n"
        x += f"    Number of samples: {self.__len__()}\n"
        x += f"    Root location: {self.root}\n"
        xr = self.extra_repr()
        xr = "\n    ".join(xr.split("\n"))
        x += f"    {xr}"
        return x
