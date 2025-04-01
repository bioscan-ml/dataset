r"""
BIOSCAN-1M PyTorch dataset.

:Date: 2024-05-20
:Authors:
    - Scott C. Lowe <scott.code.lowe@gmail.com>
:Copyright: 2024, Scott C. Lowe
:License: MIT
"""

import os
import pathlib
import warnings
import zipfile
from enum import Enum
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas
import PIL
import torch
from torchvision.datasets.utils import check_integrity, download_url
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

PARTITIONING_VERSIONS = [
    "large_diptera_family",
    "medium_diptera_family",
    "small_diptera_family",
    "large_insect_order",
    "medium_insect_order",
    "small_insect_order",
    "clibd",
]

VALID_SPLITS = ["train", "val", "test", "no_split"]
VALID_METASPLITS = ["all"]

CLIBD_PARTITIONING_DIRNAME = "CLIBD_partitioning"

CLIBD_PARTITION_ALIASES = {
    "pretrain": "no_split",
    "train": "train_seen",
    "val": "val_seen",
    "test": "test_seen",
    "key_unseen": "test_unseen_keys",
}
CLIBD_VALID_SPLITS = [
    "no_split",
    "seen_keys",
    "single_species",
    "test_seen",
    "test_unseen",
    "test_unseen_keys",
    "train_seen",
    "val_seen",
    "val_unseen",
    "val_unseen_keys",
]
CLIBD_VALID_METASPLITS = [
    "all",
    "all_keys",
    "no_split_and_seen_train",
]


class MetadataDtype(Enum):
    DEFAULT = "BIOSCAN1M_default_dtypes"


def load_bioscan1m_metadata(
    metadata_path,
    max_nucleotides=660,
    reduce_repeated_barcodes=False,
    split=None,
    partitioning_version="large_diptera_family",
    clibd_partitioning_path=None,
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
        The dataset partition. For the BIOSCAN-1M partitioning versions
        ({large/meduim/small}_{diptera_family/insect_order}), this
        should be one of:

        - ``"train"``
        - ``"val"``
        - ``"test"``
        - ``"no_split"``

        For the CLIBD partitioning version, this should be one of:

        - ``"all_keys"`` (the keys are used as a reference set for retreival tasks)
        - ``"no_split"`` (equivalent to ``"pretrain"`` in BIOSCAN-5M; labels here are not to species level)
        - ``"no_split_and_seen_train"`` (used for model training)
        - ``"seen_keys"``
        - ``"single_species"``
        - ``"test_seen"``
        - ``"test_unseen"``
        - ``"test_unseen_keys"``
        - ``"train_seen"``
        - ``"val_seen"``
        - ``"val_unseen"``
        - ``"val_unseen_keys"``
        - Additionally, :class:`~bioscan_dataset.BIOSCAN5M` split names are accepted as
          aliases for the corresponding CLIBD partitions.

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
        - ``"clibd"``

        The ``"clibd"`` partitioning version was introduced by the paper
        `CLIBD: Bridging Vision and Genomics for Biodiversity Monitoring at Scale
        <https://arxiv.org/abs/2405.17537>`__, whilst the other partitions were
        introduced in the `BIOSCAN-1M paper <https://arxiv.org/abs/2307.10455>`__.

        To use the CLIBD partitioning, download and extract the partition files from
        `here <https://huggingface.co/datasets/bioscan-ml/bioscan-clibd/resolve/335f24b/data/BIOSCAN_1M/CLIBD_partitioning.zip>`__.

        .. versionchanged:: 1.2.0
            Added support for CLIBD partitioning.

    clibd_partitioning_path : str, optional
        Path to the CLIBD_partitioning directory. By default, this is a subdirectory
        named ``"CLIBD_partitioning"`` in the directory containing ``metadata_path``.

    **kwargs
        Additional keyword arguments to pass to :func:`pandas.read_csv`.

    Returns
    -------
    df : pandas.DataFrame
        The metadata DataFrame.
        If the CLIBD partitioning files are present, the DataFrame will contain an
        additional column named ``"clibd_split"`` which indicates the CLIBD split for
        each sample.
    """  # noqa: E501
    if dtype == MetadataDtype.DEFAULT:
        # Use our default column data types
        dtype = COLUMN_DTYPES
    partitioning_version = partitioning_version.lower()

    # Handle CLIBD partitioning path
    explicit_clibd_partitioning_path = clibd_partitioning_path is not None
    if clibd_partitioning_path is None:
        clibd_partitioning_path = os.path.join(os.path.dirname(metadata_path), CLIBD_PARTITIONING_DIRNAME)
    if os.path.isdir(clibd_partitioning_path):
        pass
    elif partitioning_version == "clibd":
        raise EnvironmentError(
            f"{partitioning_version} partitioning requested, but the corresponding"
            f" partitioning data could not be found at: {repr(clibd_partitioning_path)}"
        )
    elif explicit_clibd_partitioning_path:
        raise EnvironmentError(
            f"The CLIBD partitioning data could not be found at the specified path: {repr(clibd_partitioning_path)}"
        )
    else:
        clibd_partitioning_path = None

    if partitioning_version == "clibd":
        # Handle BIOSCAN-5M partition names as aliases for CLIBD partitions
        split = CLIBD_PARTITION_ALIASES.get(split, split)

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
    # Add clibd_split column, indicating splits for CLIBD
    if clibd_partitioning_path is not None and (
        partitioning_version != "clibd" or split is None or split in CLIBD_VALID_METASPLITS
    ):
        split_data = []
        for p in CLIBD_VALID_SPLITS:
            _split = pandas.read_csv(os.path.join(clibd_partitioning_path, f"{p}.txt"), names=["sampleid"])
            _split["clibd_split"] = p
            split_data.append(_split)
        split_data = pandas.concat(split_data)
        df = pandas.merge(df, split_data, on="sampleid", how="left")
        # Check that all samples have a clibd_split value
        if df["clibd_split"].isna().any():
            raise RuntimeError(
                "Some samples in the metadata file were not assigned a clibd_split value."
                " Please check that the partitioning files are present and correctly formatted."
            )
    # Filter to just the split of interest
    if split is None or split == "all":
        pass
    elif partitioning_version == "clibd":
        try:
            partition = pandas.read_csv(os.path.join(clibd_partitioning_path, f"{split}.txt"), names=["sampleid"])
        except FileNotFoundError:
            if split not in CLIBD_VALID_METASPLITS + CLIBD_VALID_SPLITS:
                raise ValueError(
                    f"Invalid split value: {repr(split)}. Valid splits for partitioning version"
                    f" {repr(partitioning_version)} are:"
                    f" {', '.join(repr(s) for s in CLIBD_VALID_METASPLITS + CLIBD_VALID_SPLITS)}"
                ) from None
            raise
        # Use the order of samples from the CLIBD partitioning files
        df = pandas.merge(partition, df, on="sampleid", how="left")
        if "clibd_split" not in df.columns:
            # Don't overwrite the clibd_split column if it already exists due to use of a metasplit.
            # Otherwise, add the clibd_split column now.
            df["clibd_split"] = split
    elif split in VALID_SPLITS:
        try:
            select = df[partitioning_version] == split
        except KeyError:
            if partitioning_version not in PARTITIONING_VERSIONS:
                raise ValueError(
                    f"Invalid partitioning version: {repr(partitioning_version)}."
                    f" Valid partitioning versions are: {', '.join(repr(s) for s in PARTITIONING_VERSIONS)}"
                ) from None
            raise
        df = df.loc[select]
    else:
        raise ValueError(
            f"Invalid split value: {repr(split)}. Valid splits for partitioning version"
            f" {repr(partitioning_version)} are: {', '.join(repr(s) for s in VALID_METASPLITS + VALID_SPLITS)}"
        )
    return df


load_metadata = load_bioscan1m_metadata


def extract_zip_without_prefix(
    from_path: Union[str, pathlib.Path],
    to_path: Optional[Union[str, pathlib.Path]] = None,
    drop_prefix: Optional[str] = None,
    remove_finished: bool = False,
):
    r"""
    Extract a zip file, optionally modifying the output paths by dropping a parent directory.

    Parameters
    ----------
    from_path : str
        Path to the zip file to be extracted.
    to_path : str
        Path to the directory the file will be extracted to.
        If omitted, the directory of the file is used.
    drop_prefix : str, optional
        Removes a prefix from the paths in the zip file.
    remove_finished : bool, default=False
        If ``True``, remove the file after the extraction.
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, "r") as h_zip:
        for member in h_zip.namelist():
            output_path = member
            # If drop_prefix is specified, remove it from the output path
            if drop_prefix is not None and output_path.startswith(drop_prefix):
                output_path = member[len(drop_prefix) :]
                output_path = output_path.lstrip(os.sep + r"/")
            # Construct the full output path
            output_path = os.path.join(to_path, output_path)
            # Check if the member is a directory
            if member.endswith(os.sep) or member.endswith("/"):
                os.makedirs(output_path, exist_ok=True)
                continue
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Extract the file
            with h_zip.open(member) as source, open(output_path, "wb") as target:
                target.write(source.read())

    if remove_finished:
        os.remove(from_path)


class BIOSCAN1M(VisionDataset):
    r"""`BIOSCAN-1M <https://github.com/bioscan-ml/BIOSCAN-1M>`__ Dataset.

    Parameters
    ----------
    root : str
        The root directory, to contain the downloaded tarball file, and
        the image directory, BIOSCAN-1M.

    split : str, default="train"
        The dataset partition. For the BIOSCAN-1M partitioning versions
        ({large/medium/small}_{diptera_family/insect_order}), this
        should be one of:

        - ``"train"``
        - ``"val"``
        - ``"test"``
        - ``"no_split"``

        For the CLIBD partitioning version, this should be one of:

        - ``"all_keys"`` (keys are used as a reference set for retreival tasks)
        - ``"no_split"`` (similar to pretrain in BIOSCAN-5M, this has incomplete labels)
        - ``"no_split_and_seen_train"`` (used for model training)
        - ``"seen_keys"``
        - ``"single_species"``
        - ``"test_seen"``
        - ``"test_unseen"``
        - ``"test_unseen_keys"``
        - ``"train_seen"``
        - ``"val_seen"``
        - ``"val_unseen"``
        - ``"val_unseen_keys"``
        - Additionally, :class:`~bioscan_dataset.BIOSCAN5M` split names are accepted as
          aliases for the corresponding CLIBD partitions.

        If ``split`` is ``None`` or ``"all"``, the data is not filtered by
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
        - ``"clibd"``

        The ``"clibd"`` partitioning version was introduced by the paper
        `CLIBD: Bridging Vision and Genomics for Biodiversity Monitoring at Scale
        <https://arxiv.org/abs/2405.17537>`__, whilst the other partitions were
        introduced in the `BIOSCAN-1M paper <https://arxiv.org/abs/2307.10455>`__.

        To use the CLIBD partitioning, download and extract the partition files from
        `here <https://huggingface.co/datasets/bioscan-ml/bioscan-clibd/resolve/335f24b/data/BIOSCAN_1M/CLIBD_partitioning.zip>`__
        into the ``root`` directory.

        .. versionchanged:: 1.2.0
            Added support for CLIBD partitioning.

    modality : str or Iterable[str], default=("image", "dna")
        Which data modalities to use. One of, or a list of:
        ``"image"``, ``"dna"``, or any column name in the metadata TSV file.

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

    download : bool, default=False
        If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.
        Images are only downloaded if the ``"image"`` modality is requested.
        Note that only ``image_package`` values ``"cropped_256"`` and ``"original_256"``
        are currently supported for automatic image download.

        .. versionadded:: 1.2.0
    """  # noqa: E501

    base_folder = "bioscan1m"
    meta = {
        "urls": [
            "https://zenodo.org/records/8030065/files/BIOSCAN_Insect_Dataset_metadata.tsv",
            "https://huggingface.co/datasets/bioscan-ml/BIOSCAN-1M/resolve/33e1f31/BIOSCAN_Insect_Dataset_metadata.tsv",
        ],
        "filename": "BIOSCAN_Insect_Dataset_metadata.tsv",
        "csv_md5": "dec3bb23870a35e2e13bc17a5809c901",
    }
    zip_files = {
        "cropped_256": {
            "url": "https://zenodo.org/records/8030065/files/cropped_256.zip",
            "md5": "fe1175815742db14f7372d505345284a",
        },
        "original_256": {
            "url": "https://zenodo.org/records/8030065/files/original_256.zip",
            "md5": "9729fc1c49d84e7f1bfc6f5a0916d72b",
        },
    }
    image_files = [
        (
            "part18/5351601.jpg",
            {"cropped_256": "f8d7afc0dd02404863d55882d848f5cf", "original_256": "9349153e047725e4623d706a97deec66"},
        ),
        (
            "part93/BIOUG73231-D12.jpg",
            {"cropped_256": "5b60309d997570052003dc50d4d75105", "original_256": "91f5041d6b9fbacfa9c7a4d4d7250bde"},
        ),
        (
            "part99/BIOUG88809-E11.jpg",
            {"cropped_256": "a1def67aea11a051c1c7fb8d0cab76c0", "original_256": "17e74a4691e0010b8d3d80a75b9a8bbd"},
        ),
        (
            "part113/BIOUG79013-C04.jpg",
            {"cropped_256": "b1c1df1b22aee1a52a10ea3bc9ce9d23", "original_256": "0d01d3818610460850396b6dce0fdc2b"},
        ),
    ]

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

        self.metadata = None
        self.root = root
        self.image_package = image_package
        # New file structure from versions >=1.2.0
        self.metadata_path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if (
            not os.path.isdir(os.path.join(self.root, self.base_folder))
            and os.path.isfile(os.path.join(self.root, self.meta["filename"]))
            and os.path.isdir(os.path.join(self.root, "bioscan"))
        ):
            # Old file structure from versions <=1.1.0
            self.base_folder = "bioscan"
            self.metadata_path = os.path.join(self.root, self.meta["filename"])
        self.image_dir = os.path.join(self.root, self.base_folder, "images", self.image_package)

        self.partitioning_version = partitioning_version.lower()
        if self.partitioning_version == "clibd":
            self.split = CLIBD_PARTITION_ALIASES.get(split, split)
        else:
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

        # Check that the target_type is compatible with the partitioning version
        if self.partitioning_version == "clibd":
            too_fine_ranks = set()
        else:
            too_fine_ranks = {"subfamily", "tribe", "genus", "species"}
        if self.partitioning_version in {"large_insect_order", "medium_insect_order", "small_insect_order"}:
            too_fine_ranks.add("family")
        bad_ranks = too_fine_ranks.intersection(self.target_type)
        if bad_ranks:
            warnings.warn(
                f"The target_type includes taxonomic ranks {bad_ranks} that are more"
                f" fine-grained than the partitioning version ('{self.partitioning_version}')"
                " was designed for."
                " This will mean the test partition contains categories which do not"
                " appear in the train partition.",
                UserWarning,
                stacklevel=2,
            )

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if self.target_format not in ["index", "text"]:
            raise ValueError(f"Unknown target_format: {repr(self.target_format)}")

        if download:
            self.download()

        if not self._check_integrity():
            raise EnvironmentError(f"{type(self).__name__} dataset not found in {self.root}.")

        self._load_metadata()

    def index2label(
        self,
        index: Union[int, npt.ArrayLike],
        column: Optional[str] = None,
    ) -> Union[str, npt.NDArray[np.str_]]:
        r"""
        Convert target's integer index to text label.

        .. versionadded:: 1.1.0

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

        .. versionadded:: 1.1.0

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
            Entries containing missing values, indicated by empty strings, are mapped
            to ``-1``.
        """
        if column is not None:
            pass
        elif len(self.target_type) == 1:
            column = self.target_type[0]
        else:
            raise ValueError("column must be specified if there isn't a single target_type")
        if isinstance(label, str):
            # Single index
            if label == "":
                return -1
            try:
                return self.metadata[column].cat.categories.get_loc(label)
            except KeyError:
                raise KeyError(f"Label {repr(label)} not found in metadata column {repr(column)}") from None
        labels = label
        try:
            out = [-1 if lab == "" else self.metadata[column].cat.categories.get_loc(lab) for lab in labels]
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
        image : PIL.Image.Image or Any
            The image, if the ``"image"`` modality is requested, optionally transformed
            by the ``transform`` pipeline.

        dna : str or Any
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
                raise ValueError(f"Unfamiliar modality: {repr(modality)}")
            values.append(X)

        target = []
        for t in self.target_type:
            if self.target_format == "index":
                target.append(sample[f"{t}_index"])
            elif self.target_format == "text":
                target.append(sample[t])
            else:
                raise ValueError(f"Unknown target_format: {repr(self.target_format)}")

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        values.append(target)
        return tuple(values)

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

    def _check_integrity_images(self, verbose=1) -> bool:
        check_all = True
        for file, data in self.image_files:
            file = os.path.join(self.image_dir, file)
            if self.image_package in data:
                check = check_integrity(file, data[self.image_package])
            else:
                check = os.path.exists(file)
            if verbose >= 1 and not check:
                if not os.path.exists(file):
                    print(f"File missing: {file}")
                else:
                    print(f"File invalid: {file}")
            if verbose >= 2 and check:
                print(f"File present: {file}")
            check_all &= check
        return check_all

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
        if "image" in self.modality:
            check &= self._check_integrity_images(verbose=verbose)
        return check

    def _download_metadata(self, verbose=1) -> None:
        if self._check_integrity_metadata(verbose=verbose):
            if verbose >= 1:
                print("Metadata CSV file already downloaded and verified")
            return
        download_url(
            self.meta["urls"][0],
            root=os.path.dirname(self.metadata_path),
            filename=os.path.basename(self.metadata_path),
            md5=self.meta["csv_md5"],
        )

    def _download_images(self, remove_finished=False, verbose=1) -> None:
        if self._check_integrity_images(verbose=verbose):
            if verbose >= 1:
                print("Images already downloaded and verified")
            return
        if self.image_package not in self.zip_files:
            raise NotImplementedError(
                f"Automatic download of image_package={repr(self.image_package)} is not yet implemented."
                " Please manually download and extract the zip files."
            )
        data = self.zip_files[self.image_package]
        filename = os.path.basename(data["url"])
        download_url(data["url"], self.root, filename=filename, md5=data.get("md5"))
        archive = os.path.join(self.root, filename)
        extract_zip_without_prefix(
            archive,
            os.path.join(self.root, self.base_folder),
            drop_prefix="bioscan",
            remove_finished=remove_finished,
        )

    def download(self) -> None:
        r"""
        Download and extract the data.

        .. versionadded:: 1.2.0
        """
        self._download_metadata()
        if "image" in self.modality:
            self._download_images()

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
            usecols=USECOLS + [p for p in PARTITIONING_VERSIONS if p != "clibd"],
        )
        return self.metadata
