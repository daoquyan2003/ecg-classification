import ast
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from torch.utils.data import Dataset

# import datamodule


class PTBDataset(Dataset):
    """`Dataset` for the PTB-XL dataset.

    This class loads the dataset, apply the preprocessing steps if given, and returns the data in
    the format.
    """

    def __init__(self, data_dir: str, scaler: StandardScaler | None) -> None:
        super().__init__()

        # Read csv files
        self.data_dir = data_dir
        self.ptb_df = pd.read_csv(
            os.path.join(self.data_dir, "ptbxl_database.csv"), index_col="ecg_id"
        )
        self.agg_df = pd.read_csv(os.path.join(self.data_dir, "scp_statements.csv"), index_col=0)

        self.scaler = scaler

        # Read wfdb files
        data = np.array(
            [wfdb.rdsamp(os.path.join(self.data_dir, f))[0] for f in self.ptb_df.filename_lr]
        )
        # Convert from string to dict
        self.ptb_df.scp_codes = self.ptb_df.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Filter out columns where diagnostic != 1
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]

        def agg(y_dict: dict):
            """This function takes a dictionary `y_dict` as input and returns a list of unique
            diagnostic classes. The function iterates over the keys in `y_dict` and checks if each
            key is present in the index of `self.agg_df`. If a key is found, it retrieves the
            corresponding diagnostic class from `self.agg_df` and checks if it is not equal to
            "nan". If the diagnostic class is not "nan", it is added to the `tmp` list. Finally,
            the function returns a list of unique diagnostic classes by converting `tmp` to a set
            and back to a list.

            Parameters:
                y_dict (dict): A dictionary containing keys to be checked in `self.agg_df`.

            Returns:
                list: A list of unique diagnostic classes.
            """
            tmp = []

            for key in y_dict.keys():
                if key in self.agg_df.index:
                    c = self.agg_df.loc[key].diagnostic_class
                    if str(c) != "nan":
                        tmp.append(c)
            return list(set(tmp))

        self.ptb_df["diagnostic_superclass"] = self.ptb_df.scp_codes.apply(agg)
        self.ptb_df["superdiagnostic_len"] = self.ptb_df["diagnostic_superclass"].apply(
            lambda x: len(x)
        )
        counts = pd.Series(np.concatenate(self.ptb_df.diagnostic_superclass.values)).value_counts()
        self.ptb_df["diagnostic_superclass"] = self.ptb_df["diagnostic_superclass"].apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))  # noqa
        )

        X_data = data[self.ptb_df["superdiagnostic_len"] >= 1]
        Y_data = self.ptb_df[self.ptb_df["superdiagnostic_len"] >= 1]

        mlb = MultiLabelBinarizer()
        mlb.fit(Y_data["diagnostic_superclass"].values)

        self.Y = mlb.transform(Y_data["diagnostic_superclass"].values)

        if self.scaler is not None:
            self.X = self.apply_scaler(X_data, self.scaler)
        else:
            self.X = X_data

        del X_data, Y_data, counts, data

        assert len(self.X) == len(self.Y), "X and Y have unmatched lengths."

    def __len__(self) -> int:
        """Returns the number of elements in the dataset.

        Returns:
            int: The number of elements in the dataset.
        """
        return len(self.Y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the item at the specified index in the dataset.

        Parameters:
        idx (int): The index of the item to retrieve.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input tensor and the corresponding label tensor.
        """
        x_tensor = torch.from_numpy(self.X[idx])
        y_tensor = torch.from_numpy(self.Y[idx])
        return x_tensor, y_tensor

    def apply_scaler(self, inputs: np.array, scaler) -> np.array:
        """Applies standardization to each individual ECG signal.

        Parameters
        ----------
        inputs: np.array
            Array of ECG signals.
        scaler: StandardScaler
            Standard scaler object.

        Returns
        -------
        np.array
            Array of standardized ECG signals.
        """

        temp = []
        for x in inputs:
            x_shape = x.shape
            temp.append(scaler.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
        temp = np.array(temp)
        return temp


if __name__ == "__main__":
    dataset = PTBDataset(data_dir="./data/ptb", scaler=None)
    for i, (x, y) in enumerate(dataset):
        print(x)
        print(y)
        print(x.shape, y.shape)

        print(type(x), type(y))
        if i == 1:
            break
