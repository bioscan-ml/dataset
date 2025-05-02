"""
Canadian Invertebrate 1.5M PyTorch Dataset.

"""
# Necessary Libraries
import os
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas
import PIL
import torch
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torch.utils.data import Dataset # Imported Dataset from pytorch

# Column Types based on the pretrained meta data, subject to change based on how the data will be organized.  
COLUMN_DTYPES = {
    "processid": str,
    "sampleid": str,
    "bin_uri": str,
    "phylum": "category",
    "class": "category",
    "order": "category",
    "family": "category",
    "genus": "category",
    "species": "category",
    "dna_seq": str,
    "seq_len": float,
}

USECOLS = [
    "processid",
    "sampleid",
    "bin_uri",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "dna_seq",
    "seq_len",
]

# Alias Names 
VALID_SPLITS = ["pretrain", "train", "val", "test", "key_unseen", "val_unseen", "test_unseen", "other_heldout"]
SPLIT_ALIASES = {"validation": "val"}
VALID_METASPLITS = ["all", "seen", "unseen"]
SEEN_SPLITS = ["train", "val", "test"]
UNSEEN_SPLITS = ["key_unseen", "val_unseen", "test_unseen"]


# Taken Directly from Bioscan5M
def explode_metasplit(metasplit: str, verify: bool = False) -> Set[str]:
    
    r"""
    Convert a metasplit string into its set of constituent splits.

    .. versionadded:: 1.2.0

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
    DEFAULT = "CanadianInvertebrate_default_dtypes" # Take the defaults in the Canadian Invertebrate Database
    
    
def load_CanadianInvertebrate_metadata(Enum):
    """
    Load Canadian Invertebrate 1.5M data from csv file or the preprocessed data to prepare it for training
    
    This function help with sorting alias when others are inputing the dataset for training
    """
    
    return 0

class CanadianInvertebrates(Dataset):
    
    r"""
    
    
    """
    base_folder = "CanadianInvertebrates"
    
    meta = {
        "urls": [
            ""
        ],
        "filename": os.path.join("metadata", "csv", "CanadianInvertebrate1_5M.csv"),
        "archive_md5": " ", # hash
        "csv_md5": " ",
    }
    zip_files = {
        "eval": {
            "cropped_256": {
                "url": " ",
                "md5": " ",
            },
        },
        "pretrain01": {
            "cropped_256": {
                "url": " ",
                "md5": " ",
            },
        },
        "pretrain02": {
            "cropped_256": {
                "url": " ",
                "md5": " ",
            },
        },
        "train": {
            "cropped_256": {
                "url": " ",
                "md5": " ",
            },
        },
    }

    def __init__():
        
        """ 
        Needed to set up dataset, load data from file 
        """
        
    def __len__(self):
        """
        Returns the number of samples
        
        """
    
    def label2index():
        
        return 0
    
    def index2label():
        
        return 0
    
    def __getitem__(self,index:int) -> Tuple[Any,...]:
        
        """
        Get a sample from the dataset
        
        """
        
    def _check_integrity_metadata(self, verbose=1) -> bool:
        return 0
    
    def _check_integrity(self, verbose=1) -> bool:
        
        """
        Used to check if the data is already downloaded and extracted
        
        """

    def _download_metadata(self, verbose=1) -> None:
        """
        Check if metadata is downloaded
        """

    
    def download(self) -> None:
        """
        Download and extract the data.
        """
    

    def _load_metadata(self) -> pandas.DataFrame:
        """
        Load metadata from CSV file and prepare it for training.
        """
  

    def extra_repr(self) -> str:
        
        
        return 0