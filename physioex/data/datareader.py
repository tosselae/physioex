import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
import torch
from loguru import logger
from ml_dtypes import bfloat16


class Reader(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def get_table(self):
        pass


DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


class MemmapReader(Reader):
    # this object abstracts the reading of subject data from memmap files for a specific dataset
    def __init__(
        self,
        data_folder: str,
        dataset: str,
        preprocessing: str,
        sequence_length: int,
        channels_index: List[int],
        offset: int,
    ):
        self.preprocessing = preprocessing
        self.dataset = dataset
        self.data_path = os.path.join(data_folder, dataset, preprocessing)
        self.labels_path = os.path.join(data_folder, dataset, "labels")

        self.L = sequence_length
        self.channels_index = channels_index
        self.offset = offset

        # get the scaling parameters
        scaling = np.load(os.path.join(self.data_path, "scaling.npz"))

        self.input_shape = list(scaling["mean"].shape)

        self.mean = torch.tensor(scaling["mean"][channels_index]).to( DTYPE )
        self.std = torch.tensor(scaling["std"][channels_index]).to( DTYPE )

        # read the table
        self.table = pd.read_csv(os.path.join(data_folder, dataset, "table.csv"))

        num_windows = self.table["num_windows"].values
        subjects_id = self.table["subject_id"].values

        if np.max(num_windows) < self.L:
            logger.warning(
                f"Sequence length {self.L} is greater than the max number of windows {np.max(num_windows)} for dataset {dataset}."
            )

        self.len = num_windows - self.L

        neg = np.where(self.len < 0)[0]
        if len(neg) > 0:
            self.len[neg] = 0

        self.len = int(np.sum(self.len + 1))

        self.subject_idx, self.relative_idx, self.windows_index = build_index(
            num_windows, subjects_id, self.L
        )

    def get_n_subjects( self ):
        return len( np.unique( self.table["subject_id"].values ) )

    def get_table(self):
        folds_colum = [col for col in self.table.columns if "fold_" in col]
        return self.table[["subject_id", "num_windows"] + folds_colum].copy()

    def __len__(self):
        return self.len

    def get_signal(self, idx):
        idx = idx - self.offset

        relative_id = self.relative_idx[idx]
        subject_id = self.subject_idx[idx]
        num_windows = self.windows_index[subject_id]

        input_shape = tuple([num_windows] + self.input_shape)

        data_path = os.path.join(self.data_path, str(subject_id) + ".npy")

        X = np.memmap(data_path, dtype=bfloat16, mode="r", shape=input_shape)

        if relative_id + self.L > num_windows:
            X = X[relative_id:, self.channels_index]

            remainer = self.L - X.shape[0]
            # add zeros to the end of the array
            X = np.concatenate([X, np.zeros((remainer, *X.shape[1:]))], axis=0)

        else:
            X = X[relative_id : relative_id + self.L, self.channels_index]

        # X = (torch.tensor(X).float() - self.mean) / self.std

        # tensor conversion
        X = torch.from_numpy(X.astype(np.float32)).to(DTYPE)

        # invert scale
        X = X * self.std + self.mean


        return X

    def get_stages(self, idx):
        idx = idx - self.offset

        relative_id = self.relative_idx[idx]
        subject_id = self.subject_idx[idx]
        num_windows = self.windows_index[subject_id]

        labels_shape = (num_windows,)

        labels_path = os.path.join(self.labels_path, str(subject_id) + ".npy")

        y = np.memmap(labels_path, dtype="int16", mode="r", shape=labels_shape)

        if relative_id + self.L > num_windows:
            y = y[relative_id:]

            remainer = self.L - len(y)
            # associate the padded values to the class 6
            y = np.concatenate([y, np.ones(remainer) * 5], axis=0)
        else:
            y = y[relative_id : relative_id + self.L]

        y = torch.tensor(y).long()

        return y, subject_id

    def __getitem__(self, idx):
        X = self.get_signal(idx)
        y, subject_id = self.get_stages(idx)

        return X, y, subject_id

class WholeNightReader(MemmapReader):
    # suppose the batch_size to be 1
    # reads every time one night
    def __init__(
        self,
        data_folder: str,
        dataset: str,
        preprocessing: str,
        # sequence_length: int,
        channels_index: List[int],
        offset: int,
    ):
        super().__init__(
            data_folder=data_folder,
            dataset=dataset,
            preprocessing=preprocessing,
            sequence_length=30 * 2 * 60 * 24,  # 24 hours,
            channels_index=channels_index,
            offset=offset,
        )

    def get_signal(self, idx):
        idx = idx - self.offset

        subject_id = self.subject_idx[idx]
        num_windows = self.windows_index[subject_id]

        input_shape = tuple([num_windows] + self.input_shape)

        data_path = os.path.join(self.data_path, str(subject_id) + ".npy")

        X = np.memmap(data_path, dtype=bfloat16, mode="r", shape=input_shape)
        X = X[:, self.channels_index]
        # X = (torch.tensor(X).float() - self.mean) / self.std

        X = torch.from_numpy(X.astype(np.float32)).to(DTYPE)

        # invert scale
        X = X * self.std + self.mean

        return X

    def get_stages(self, idx):
        idx = idx - self.offset

        subject_id = self.subject_idx[idx]
        num_windows = self.windows_index[subject_id]

        labels_shape = (num_windows,)

        labels_path = os.path.join(self.labels_path, str(subject_id) + ".npy")

        y = np.memmap(labels_path, dtype="int16", mode="r", shape=labels_shape)

        y = torch.tensor(y).long()

        return y, subject_id


class AgeMemmapReader(MemmapReader):
    def __init__(
        self,
        data_folder: str,
        dataset: str,
        preprocessing: str,
        sequence_length: int,
        channels_index: List[int],
        offset: int,
    ):
        self.preprocessing = preprocessing

        self.data_path = os.path.join(data_folder, dataset, preprocessing)
        self.labels_path = os.path.join(data_folder, dataset, "labels")

        self.L = sequence_length
        self.channels_index = channels_index
        self.offset = offset

        # get the scaling parameters
        scaling = np.load(os.path.join(self.data_path, "scaling.npz"))

        self.input_shape = list(scaling["mean"].shape)

        self.mean = torch.tensor(scaling["mean"][channels_index]).to( DTYPE )
        self.std = torch.tensor(scaling["std"][channels_index]).to( DTYPE )

        # read the table
        self.table = pd.read_csv(os.path.join(data_folder, dataset, "table.csv"))

        ## Modify table to drop the wrong age
        self.table = self.clean_and_update_df(self.table)

        num_windows = self.table["num_windows"].values
        subjects_id = self.table["subject_id"].values

        if np.max(num_windows) < self.L:
            logger.warning(
                f"Sequence length {self.L} is greater than the max number of windows {np.max(num_windows)} for dataset {dataset}."
            )

        self.len = num_windows - self.L

        neg = np.where(self.len < 0)[0]
        if len(neg) > 0:
            self.len[neg] = 0

        self.len = int(np.sum(self.len + 1))

        self.subject_idx, self.relative_idx, self.windows_index = build_index(
            num_windows, subjects_id, self.L
        )

    def __getitem__(self, idx):
        idx = idx - self.offset
        subject_id = self.subject_idx[idx]
        age = self.table.loc[
            self.table["subject_id"] == subject_id, "nsrr_age"
        ].values.astype(float)[0]

        y = torch.tensor([age], dtype=torch.float32)

        X = self.get_signal(idx)

        return X, y

    def clean_and_update_df(self, df):
        "Reads the table of the dataset and removes entries with NaN age"

        # Remove entries where nsrr_age is missing
        df = df.dropna(subset=["nsrr_age"]).reset_index(drop=True)

        # Recalculate start_index and end_index
        df["start_index"] = 0
        df["end_index"] = df["num_windows"].cumsum()
        df["start_index"] = df["end_index"].shift(1, fill_value=0)

        return df


class DataReader(Reader):
    def __init__(
        self,
        data_folder: str,
        dataset: str,
        preprocessing: str,
        sequence_length: int,
        channels_index: List[int],
        offset: int,
        hpc: bool,
        task: str = "sleep",
    ):
        if task == "sleep" and sequence_length > 0:
            self.reader = MemmapReader(
                data_folder=data_folder,
                dataset=dataset,
                preprocessing=preprocessing,
                sequence_length=sequence_length,
                channels_index=channels_index,
                offset=offset,
                )
        elif task == "sleep" and sequence_length == -1:
            self.reader = WholeNightReader(
                data_folder=data_folder,
                dataset=dataset,
                preprocessing=preprocessing,
                channels_index=channels_index,
                offset=offset,
            )
        elif task == "age":
            self.reader = AgeMemmapReader(
                data_folder=data_folder,
                dataset=dataset,
                preprocessing=preprocessing,
                sequence_length=sequence_length,
                channels_index=channels_index,
                offset=offset,
            )
        else:
            raise ValueError("task must be either sleep or age")

    def __len__(self):
        return self.reader.__len__()

    def __getitem__(self, idx):
        return self.reader[idx]

    def __del__(self):
        if hasattr(self, "reader"):
            del self.reader

    def get_table(self):
        return self.reader.get_table()

    def get_sequence_length(self):
        return self.reader.L


def build_index(nums_windows, subjects_ids, sequence_length):
    nums_windows = nums_windows.astype(int)

    data_len = nums_windows - sequence_length
    neg = np.where(data_len < 0)[0]
    if len(neg) > 0:
        data_len[neg] = 0

    data_len = np.sum(data_len + 1)

    subject_idx = np.zeros(data_len, dtype=np.int16)
    relative_idx = np.zeros(data_len, dtype=np.int16)

    start_index = 0

    for i, (num_windows, subject_id) in enumerate(zip(nums_windows, subjects_ids)):
        # check if the subject_id or the relative_id can be stored in a uint16
        if subject_id > np.iinfo(np.int16).max:
            raise ValueError(
                f"subject_id {subject_id} exceeds the maximum value for np.int16"
            )

        num_sequences = max((num_windows - sequence_length, 0)) + 1

        if num_sequences > np.iinfo(np.int16).max:
            raise ValueError(
                f"Relative index {num_sequences} exceeds the maximum value for np.int16"
            )

        subject_idx[start_index : start_index + num_sequences] = subject_id
        relative_idx[start_index : start_index + num_sequences] = np.arange(
            num_sequences
        )

        start_index += num_sequences

    windows_index = np.zeros(np.max(subjects_ids) + 1, dtype=np.int16)
    for i, num_windows in enumerate(nums_windows):
        windows_index[subjects_ids[i]] = num_windows

    return subject_idx, relative_idx, windows_index
