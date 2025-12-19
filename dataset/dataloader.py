import h5py
import torch
from torch.utils.data import Dataset
from model.model_s1_s2 import TransformerAE

class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, return_coords=False, da=True, s1_s2 = False):
        """
        PyTorch Dataset for HDF5 data.

        Args:
            h5_file_path (str): Path to the HDF5 file.
            return_coords (bool): If True, return coordinate arrays instead of time_gaps.
        """
        self.s1_s2 = s1_s2
        self.h5_file_path = h5_file_path
        self.return_coords = return_coords
        self.h5_file = h5py.File(h5_file_path, "r", swmr=True)

        # Always present
        self.data = self.h5_file["data"]
        self.mask = self.h5_file["mask"]

        if self.s1_s2:
            self.time_gaps_s1 = self.h5_file["time_gaps_s1"]
            self.time_gaps_s2 = self.h5_file["time_gaps_s2"]
            self.time_gaps_c = self.h5_file["time_gaps_c"]
        else: self.time_gaps = self.h5_file["time_gaps"]

        self.da = da

        # Optional returns
        if return_coords:
            self.coord_time = self.h5_file["coord_time"]
            self.coord_x = self.h5_file["coord_x"]
            self.coord_y = self.h5_file["coord_y"]

        self.length = self.data.shape[0]

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.length

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            - If return_coords is False:
                tuple: (data, time_gaps, mask)
            - If return_coords is True:
                tuple: (data, mask, coord_time, coord_x, coord_y)
        """
        data = torch.tensor(self.data[index], dtype=torch.float32)
        mask = torch.tensor(self.mask[index], dtype=torch.bool)
        if self.s1_s2:
            time_gaps_s1 = torch.tensor(self.time_gaps_s1[index], dtype=torch.int32)
            time_gaps_s2 = torch.tensor(self.time_gaps_s2[index], dtype=torch.int32)
            time_gaps_c = torch.tensor(self.time_gaps_c[index], dtype=torch.int32)
            return data, mask, time_gaps_s1, time_gaps_s2, time_gaps_c
        time_gaps = torch.tensor(self.time_gaps[index], dtype=torch.int32)
        return data, time_gaps, mask

    def __del__(self):
        """Ensure the HDF5 file is properly closed when the object is deleted."""
        pass


# Usage Example:
if __name__ == "__main__":
    h5_dataset = HDF5Dataset("../val_s1_s2.h5", s1_s2=True)
    print(len(h5_dataset))#42843 #46297
    model = TransformerAE(dbottleneck=7)

    # Example: Iterate over the dataset using DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(h5_dataset, batch_size=1, shuffle=False)#, num_workers=4)

    for i, batch in enumerate(dataloader):
        data, mask, time_gaps_s1, time_gaps_s2, time_gaps_c = batch
        print(data.shape)
        print(mask.shape)
        print(time_gaps_s1.shape)
        print(time_gaps_s2.shape)
        print(time_gaps_c.shape)
        break

