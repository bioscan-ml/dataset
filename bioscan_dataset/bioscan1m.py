r"""
BIOSCAN-1M PyTorch dataset.

:Date: 2024-05-20
:Authors:
    - Scott C. Lowe <scott.code.lowe@gmail.com>
:Copyright: 2024, Scott C. Lowe
:License: MIT
"""

import os
from enum import Enum
from typing import Any, Iterable, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas
import PIL
import torch
from torchvision.datasets.vision import VisionDataset

RGB_MEAN = torch.tensor([0.72510918, 0.72891550, 0.72956181])
RGB_STDEV = torch.tensor([0.12654378, 0.14301962, 0.16103319])

COLUMN_DTYPES = {
    "sampleid": str,
    "processid": str,
    "uri": str,
    "name": "category",
    "phylum": str,
    "class": str,
    "order": str,
    "family": str,
    "subfamily": str,
    "tribe": str,
    "genus": str,
    "species": str,
    "subspecies": str,
    "nucraw": str,
    "image_file": str,
    "large_diptera_family": "category",
    "medium_diptera_family": "category",
    "small_diptera_family": "category",
    "large_insect_order": "category",
    "medium_insect_order": "category",
    "small_insect_order": "category",
    "chunk_number": "uint8",
    "copyright_license": "category",
    "copyright_holder": "category",
    "copyright_institution": "category",
    "copyright_contact": "category",
    "photographer": "category",
    "author": "category",
}

PARTITIONING_VERSIONS = [
    "large_diptera_family",
    "medium_diptera_family",
    "small_diptera_family",
    "large_insect_order",
    "medium_insect_order",
    "small_insect_order",
]

USECOLS = [
    "sampleid",
    "uri",
    "phylum",
    "class",
    "order",
    "family",
    "subfamily",
    "tribe",
    "genus",
    "species",
    "nucraw",
    "image_file",
    "chunk_number",
]


class MetadataDtype(Enum):
    DEFAULT = "BIOSCAN1M_default_dtypes"


def load_bioscan1m_metadata(
    metadata_path,
    max_nucleotides=660,
    reduce_repeated_barcodes=False,
    split=None,
    partitioning_version="large_diptera_family",
    dtype=MetadataDtype.DEFAULT,
    **kwargs,
) -> pandas.DataFrame:
    r"""
    Load BIOSCAN-1M metadata from its TSV file, and prepare it for training.

    Parameters
    ----------
    metadata_path : str
        Path to metadata file.

    max_nucleotides : int, default=660
        Maximum nucleotide sequence length to keep for the DNA barcodes.
        Set to ``None`` to keep the original data without truncation.
        Note that the barcode should only be 660 base pairs long.
        Characters beyond this length are unlikely to be accurate.

    reduce_repeated_barcodes : str or bool, default=False
        Whether to reduce the dataset to only one sample per barcode.
        If ``True``, duplicated barcodes are removed after truncating the barcodes to
        the length specified by ``max_nucleotides`` and stripping trailing Ns.
        If ``False`` (default) no reduction is performed.

    split : str, optional
        The dataset partition, one of:

        - ``"train"``
        - ``"val"``
        - ``"test"``
        - ``"no_split"``
        - ``"all"``

        If ``split`` is ``None`` or ``"all"`` (default), the data is not filtered by
        partition and the dataframe will contain every sample in the dataset.

        Note that the contents of the split depends on the value of ``partitioning_version``.
        If ``partitioning_version`` is changed, the same ``split`` value will yield
        completely different records.

    partitioning_version : str, default="large_diptera_family"
        The dataset partitioning version, one of:

        - ``"large_diptera_family"``
        - ``"medium_diptera_family"``
        - ``"small_diptera_family"``
        - ``"large_insect_order"``
        - ``"medium_insect_order"``
        - ``"small_insect_order"``

    **kwargs
        Additional keyword arguments to pass to :func:`pandas.read_csv`.

    Returns
    -------
    df : pandas.DataFrame
        The metadata DataFrame.
    """
    if dtype == MetadataDtype.DEFAULT:
        # Use our default column data types
        dtype = COLUMN_DTYPES
    df = pandas.read_csv(metadata_path, sep="\t", dtype=dtype, **kwargs)
    # Taxonomic label column names
    label_cols = [
        "phylum",
        "class",
        "order",
        "family",
        "subfamily",
        "tribe",
        "genus",
        "species",
        "uri",
    ]
    # Truncate the DNA barcodes to the specified length
    if max_nucleotides is not None:
        df["nucraw"] = df["nucraw"].str[:max_nucleotides]
    # Reduce the dataset to only one sample per barcode
    if reduce_repeated_barcodes:
        # Shuffle the data order, to avoid bias in the subsampling that could be induced
        # by the order in which the data was collected.
        df = df.sample(frac=1, random_state=0)
        # Drop duplicated barcodes
        df["nucraw_strip"] = df["nucraw"].str.rstrip("N")
        df = df.drop_duplicates(subset=["nucraw_strip"])
        df.drop(columns=["nucraw_strip"], inplace=True)
        # Re-order the data (reverting the shuffle)
        df = df.sort_index()
    # Convert missing values to NaN
    for c in label_cols:
        df.loc[df[c] == "not_classified", c] = pandas.NA
    # Fix some tribe labels which were only partially applied
    df.loc[df["genus"].notna() & (df["genus"] == "Asteia"), "tribe"] = "Asteiini"
    df.loc[df["genus"].notna() & (df["genus"] == "Nemorilla"), "tribe"] = "Winthemiini"
    df.loc[df["genus"].notna() & (df["genus"] == "Philaenus"), "tribe"] = "Philaenini"
    # Add missing genus labels
    sel = df["genus"].isna() & df["species"].notna()
    df.loc[sel, "genus"] = df.loc[sel, "species"].apply(lambda x: x.split(" ")[0])
    # Add placeholder for missing tribe labels
    sel = df["tribe"].isna() & df["genus"].notna()
    sel2 = df["subfamily"].notna()
    df.loc[sel & sel2, "tribe"] = "unassigned " + df.loc[sel, "subfamily"]
    df.loc[sel & ~sel2, "tribe"] = "unassigned " + df.loc[sel, "family"]
    # Add placeholder for missing subfamily labels
    sel = df["subfamily"].isna() & df["tribe"].notna()
    df.loc[sel, "subfamily"] = "unassigned " + df.loc[sel, "family"]
    # Convert label columns to category dtype; add index columns to use for targets
    for c in label_cols:
        df[c] = df[c].astype("category")
        df[c + "_index"] = df[c].cat.codes
    # Filter to just the split of interest
    if split is not None and split != "all":
        select = df[partitioning_version] == split
        df = df.loc[select]
    return df


load_metadata = load_bioscan1m_metadata


class BIOSCAN1M(VisionDataset):
    r"""`BIOSCAN-1M <https://github.com/bioscan-ml/BIOSCAN-1M>`_ Dataset.

    Parameters
    ----------
    root : str
        The root directory, to contain the downloaded tarball file, and
        the image directory, BIOSCAN-1M.

    split : str, default="train"
        The dataset partition, one of:

        - ``"train"``
        - ``"val"``
        - ``"test"``
        - ``"no_split"``

    partitioning_version : str, default="large_diptera_family"
        The dataset partitioning version, one of:

        - ``"large_diptera_family"``
        - ``"medium_diptera_family"``
        - ``"small_diptera_family"``
        - ``"large_insect_order"``
        - ``"medium_insect_order"``
        - ``"small_insect_order"``

    modality : str or Iterable[str], default=("image", "dna")
        Which data modalities to use. One of, or a list of:
        ``"image"``, ``"dna"``, or any column name in the metadata CSV file.

        .. versionchanged:: 1.1.0
            Added support for arbitrary modalities.

    image_package : str, default="cropped_256"
        The package to load images from. One of:
        ``"original_full"``, ``"cropped"``, ``"original_256"``, ``"cropped_256"``.

        .. versionadded:: 1.1.0

    reduce_repeated_barcodes : bool, default=False
        Whether to reduce the dataset to only one sample per barcode.

    max_nucleotides : int, default=660
        Maximum number of nucleotides to keep in the DNA barcode.
        Set to ``None`` to keep the original data without truncation (default).
        Note that the barcode should only be 660 base pairs long.
        Characters beyond this length are unlikely to be accurate.

    target_type : str or Iterable[str], default="family"
        Type of target to use. One of, or a list of:

        - ``"phylum"``
        - ``"class"``
        - ``"order"``
        - ``"family"``
        - ``"subfamily"``
        - ``"tribe"``
        - ``"genus"``
        - ``"species"``
        - ``"uri"``

        Where ``"uri"`` corresponds to the BIN cluster label.

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

        .. versionadded:: 1.1.0

    transform : Callable, default=None
        Image transformation pipeline.

    dna_transform : Callable, default=None
        DNA barcode transformation pipeline.

    target_transform : Callable, default=None
        Label transformation pipeline.
    """

    def __init__(
        self,
        root,
        split="train",
        partitioning_version="large_diptera_family",
        modality=("image", "dna"),
        image_package="cropped_256",
        reduce_repeated_barcodes=False,
        max_nucleotides=660,
        target_type="family",
        target_format="index",
        transform=None,
        dna_transform=None,
        target_transform=None,
        download=False,
    ) -> None:
        root = os.path.expanduser(root)
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            raise NotImplementedError("Download functionality not yet implemented.")

        self.metadata = None
        self.root = root
        self.metadata_path = os.path.join(self.root, "BIOSCAN_Insect_Dataset_metadata.tsv")
        self.image_package = image_package
        self.image_dir = os.path.join(self.root, "bioscan", "images", self.image_package)

        self.partitioning_version = partitioning_version
        self.split = split
        self.target_format = target_format
        self.reduce_repeated_barcodes = reduce_repeated_barcodes
        self.max_nucleotides = max_nucleotides
        self.dna_transform = dna_transform

        if isinstance(modality, str):
            self.modality = [modality]
        else:
            self.modality = list(modality)

        if isinstance(target_type, str):
            self.target_type = [target_type]
        else:
            self.target_type = list(target_type)
        self.target_type = ["uri" if t == "dna_bin" else t for t in self.target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if self.target_format not in ["index", "text"]:
            raise ValueError(f"Unknown target_format: {self.target_format}")

        if not self._check_exists():
            raise EnvironmentError(f"{type(self).__name__} dataset not found in {self.root}.")

        self._load_metadata()

    def index2label(self, column: str, index: Union[int, Iterable[int]]) -> Union[str, npt.NDArray[np.str_]]:
        r"""
        Convert target's integer index to text label.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        column : str
            The dataset column name to map. This is the same as the ``target_type``.
        index : int or Iterable[int]
            The integer index or indices to map to labels.

        Returns
        -------
        str or numpy.array[str]
            The text label or labels corresponding to the integer index or indices
            in the specified column.
            Entries containing missing values, indicated by negative indices, are mapped
            to an empty string.
        """
        if not hasattr(index, "__len__"):
            # Single index
            if index < 0:
                return ""
            return self.metadata[column].cat.categories[index]
        index = np.asarray(index)
        out = self.metadata[column].cat.categories[index]
        out = np.asarray(out)
        out[index < 0] = ""
        return out

    def label2index(self, column: str, label: Union[str, Iterable[str]]) -> Union[int, npt.NDArray[np.int_]]:
        r"""
        Convert target's text label to integer index.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        column : str
            The dataset column name to map. This is the same as the ``target_type``.
        label : str or Iterable[str]
            The text label or labels to map to integer indices.

        Returns
        -------
        int or numpy.array[int]
            The integer index or indices corresponding to the text label or labels
            in the specified column.
            Entries containing missing values, indicated by empty strings, are mapped
            to ``-1``.
        """
        if isinstance(label, str):
            # Single index
            if label == "":
                return -1
            return self.metadata[column].cat.categories.get_loc(label)
        labels = label
        out = [-1 if lab == "" else self.metadata[column].cat.categories.get_loc(lab) for lab in labels]
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
        image : PIL.Image.Image
            The image, if the ``"image"`` modality is requested, optionally transformed
            by the ``transform`` pipeline.

        dna : str
            The DNA barcode, if the ``"dna"`` modality is requested, optionally
            transformed by the ``dna_transform`` pipeline.

        *modalities : Any
            Any other modalities requested, as specified in the ``modality`` parameter.
            The data is extracted from the appropriate column in the metadata TSV file,
            without any transformations.

            .. versionadded:: 1.1.0

        target : int or Tuple[int, ...] or str or Tuple[str, ...] or None
            The target(s), optionally transformed by the ``target_transform`` pipeline.
            If ``target_format="index"``, the target(s) will be returned as integer
            indices, with missing values filled with ``-1``.
            If ``target_format="text"``, the target(s) will be returned as a string.
            If there are multiple targets, they will be returned as a tuple.
            If ``target_type`` is an empty list, the output ``target`` will be ``None``.
        """
        sample = self.metadata.iloc[index]
        img_path = os.path.join(self.image_dir, f"part{sample['chunk_number']}", sample["image_file"])
        values = []
        for modality in self.modality:
            if modality == "image":
                X = PIL.Image.open(img_path)
                if self.transform is not None:
                    X = self.transform(X)
            elif modality in ["dna_barcode", "dna", "barcode", "nucraw"]:
                X = sample["nucraw"]
                if self.dna_transform is not None:
                    X = self.dna_transform(X)
            elif modality in self.metadata.columns:
                X = sample[modality]
            else:
                raise ValueError(f"Unfamiliar modality: {modality}")
            values.append(X)

        target = []
        for t in self.target_type:
            if self.target_format == "index":
                target.append(sample[f"{t}_index"])
            elif self.target_format == "text":
                target.append(sample[t])
            else:
                raise ValueError(f"Unknown target_format: {self.target_format}")

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        values.append(target)
        return tuple(values)

    def _check_exists(self, verbose=0) -> bool:
        r"""Check if the dataset is already downloaded and extracted.

        Parameters
        ----------
        verbose : int, default=0
            Verbosity level.

        Returns
        -------
        bool
            True if the dataset is already downloaded and extracted, False otherwise.
        """
        paths_to_check = [
            self.metadata_path,
            os.path.join(self.image_dir, "part18", "4900531.jpg"),
            os.path.join(self.image_dir, "part113", "BIOUG68114-B02.jpg"),
        ]
        check_all = True
        for p in paths_to_check:
            check = os.path.exists(p)
            if verbose >= 1 and not check:
                print(f"File missing: {p}")
            if verbose >= 2 and check:
                print(f"File present: {p}")
            check_all &= check
        return check_all

    def _load_metadata(self) -> pandas.DataFrame:
        r"""
        Load metadata from TSV file and prepare it for training.
        """
        self.metadata = load_metadata(
            self.metadata_path,
            max_nucleotides=self.max_nucleotides,
            reduce_repeated_barcodes=self.reduce_repeated_barcodes,
            split=self.split,
            partitioning_version=self.partitioning_version,
            usecols=USECOLS + PARTITIONING_VERSIONS,
        )
        return self.metadata
