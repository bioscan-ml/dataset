"""
BIOSCAN-1M PyTorch dataset.

:Date: 2024-05-20
:Authors:
    - Scott C. Lowe <scott.code.lowe@gmail.com>
:Copyright: 2024, Scott C. Lowe
:License: MIT
"""

import os

import pandas as pd
import PIL
from torchvision.datasets.vision import VisionDataset

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

usecols = [
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
    "image_file",
    "chunk_number",
]


class BIOSCAN1M(VisionDataset):
    """`BIOSCAN-1M <https://biodiversitygenomics.net/1M_insects/>`_ Dataset.

    Parameters
    ----------
    root : str
        The root directory, to contain the downloaded tarball file, and
        the image directory, BIOSCAN-1M.

    split : str, default="train"
        The dataset partition, one of ``"train"``, ``"val"``, or ``"test"``.

    target_type : str, default="species"
        Type of target to use. One of:
        ``"phylum"``, ``"class"``, ``"order"``, ``"family"``, ``"subfamily"``,
        ``"tribe"``, ``"genus"``, ``"species"``, ``"uri"``.
        Where ``"uri"`` corresponds to the BIN cluster label.

    transform : Callable, default=None
        Image transformation pipeline.

    target_transform : Callable, default=None
        Label transformation pipeline.
    """

    def __init__(
        self,
        root,
        split="train",
        target_type="species",
        transform=None,
        target_transform=None,
    ) -> None:
        root = os.path.expanduser(root)
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.metadata = None
        self.root = root
        self.image_dir = os.path.expanduser(os.path.join(self.root, "bioscan", "images", "cropped_256"))

        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if not self._check_exists():
            raise EnvironmentError(f"{type(self).__name__} dataset not found in {self.root}.")

        self.metadata = self._load_metadata()
        self._partition(split)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        sample = self.metadata.iloc[index]
        img_path = os.path.join(self.image_dir, f"part{sample['chunk_number']}", sample["image_file"])
        X = PIL.Image.open(img_path)

        target = []
        for t in self.target_type:
            target.append(sample[f"{t}_index"])

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def _check_exists(self, verbose=0) -> bool:
        """Check if the dataset is already downloaded and extracted.

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
            os.path.join(self.root, "BIOSCAN_Insect_Dataset_metadata.tsv"),
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
        """Extract metadata from sample names and creates a pandas DataFrame.

        This function goes through images and extracts age, gender, race, and the date and time of the photos from
        the file names.

        Returns
        -------
        None
        """
        df = pd.read_csv(
            os.path.join(self.root, "BIOSCAN_Insect_Dataset_metadata.tsv"),
            sep="\t",
            dtype=COLUMN_DTYPES,
            usecols=usecols,
        )
        # Convert missing values to NaN
        df[df == "not_classified"] = pd.NA
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
        for c in label_cols:
            df[c] = df[c].astype("category")
            df[c + "_index"] = df[c].cat.codes
        return df

    def _partition(self, partition_name):
        if partition_name == "train":
            partition_files = ["train_seen", "test_unseen_keys"]
        elif partition_name == "val":
            partition_files = ["test_seen", "seen_keys", "test_unseen"]
        elif partition_name == "test":
            partition_files = [
                "seen_keys",
                "test_seen",
                "test_unseen",
                "test_unseen_keys",
            ]
        elif os.path.join(self.root, partition_name + ".txt"):
            partition_files = [partition_name]
        else:
            raise ValueError(f"Unrecognized partition name: {partition_name}")

        partition_samples = []
        for fname in partition_files:
            with open(os.path.join(self.root, fname + ".txt"), "r") as f:
                partition_samples += f.readlines()

        partition_samples = [x.rstrip() for x in partition_samples]
        self.metadata = self.metadata.loc[self.metadata["sampleid"].isin(partition_samples)]
