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

import pandas as pd
import PIL
import torch
from torchvision.datasets.vision import VisionDataset

RGB_MEAN = torch.tensor([0.72510918, 0.72891550, 0.72956181])
RGB_STDEV = torch.tensor([0.66364000, 0.66088159, 0.66035860])

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
) -> pd.DataFrame:
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
    df : pd.DataFrame
        The metadata DataFrame.
    """
    if dtype == MetadataDtype.DEFAULT:
        # Use our default column data types
        dtype = COLUMN_DTYPES
    df = pd.read_csv(metadata_path, sep="\t", dtype=dtype, **kwargs)
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
        df.loc[df[c] == "not_classified", c] = pd.NA
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
        ``"image"``, ``"dna"``.

    reduce_repeated_barcodes : bool, default=False
        Whether to reduce the dataset to only one sample per barcode.

    max_nucleotides : int, default=660
        Maximum number of nucleotides to keep in the DNA barcode.
        Set to ``None`` to keep the original data without truncation (default).
        Note that the barcode should only be 660 base pairs long.
        Characters beyond this length are unlikely to be accurate.

    target_type : str, default="family"
        Type of target to use. One of:

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
        reduce_repeated_barcodes=False,
        max_nucleotides=660,
        target_type="family",
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
        self.image_dir = os.path.expanduser(os.path.join(self.root, "bioscan", "images", "cropped_256"))

        self.partitioning_version = partitioning_version
        self.split = split
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

        if not self._check_exists():
            raise EnvironmentError(f"{type(self).__name__} dataset not found in {self.root}.")

        self._load_metadata()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
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
            else:
                raise ValueError(f"Unfamiliar modality: {modality}")
            values.append(X)

        target = []
        for t in self.target_type:
            target.append(sample[f"{t}_index"])

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

    def _load_metadata(self) -> pd.DataFrame:
        r"""
        Load metadata from CSV file and prepare it for training.
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
