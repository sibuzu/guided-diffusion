import math
import random

import os
import csv
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd

def load_data(
    *,
    data_dir,
    batch_size,
    stock_size,
    deterministic=False,
    quick_sampling=True,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files, all_length = _list_image_files_recursively(data_dir)
    dataset = StockDataset(
        stock_size,
        all_files,
        all_length,
        quick_sampling = quick_sampling,
    )
    print(f"dataset={len(dataset)}")
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def _save_cache(fname, files, lens):
    merged_list = list(zip(files, lens))

    # Save the merged list to a CSV file
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in merged_list:
            writer.writerow(row)

def _load_cache(fname):
    files = []
    lens = []
    with open(fname, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            files.append(row[0])
            lens.append(int(row[1]))
    return files, lens

def _list_image_files_recursively(data_dir):
    results = []
    length = []
    cachefile = os.path.join(data_dir, "_cache_.csv")
    if os.path.isfile(cachefile):
        results, length = _load_cache(cachefile)
    else:
        for entry in sorted(os.listdir(data_dir)):
            full_path = os.path.join(data_dir, entry)
            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["feather"]:
                results.append(full_path)
                df = pd.read_feather(full_path)
                length.append(len(df))
            elif os.path.isdir(full_path):
                res, lens = _list_image_files_recursively(full_path)
                results.extend(res)
                results.extend(lens)
        if len(results) > 0:
            _save_cache(cachefile, results, length)

    return results, length


class StockDataset(Dataset):
    def __init__(
        self,
        stocks_size,
        stocks_paths,
        stocks_length,
        quick_sampling=True,
    ):
        super().__init__()
        self.stocks_size = stocks_size
        self.local_stocks = stocks_paths
        self.local_length = stocks_length
        self.quick_remained = 0
        self.df = None
        self.quick_sampling = quick_sampling
        self.fname = ""

    def __len__(self):
        lens = sum(self.local_length) - len(self.local_length) * self.stocks_size
        return lens

    def __getitem__(self, idx):
        if self.quick_sampling and self.quick_remained > 0:
            # This is dirty quick method, for the same df, we sample multiple times instead just once
            self.quick_remained = self.quick_remained - 1
            idx = random.randint(0, len(self.df) - self.stocks_size)
        else:
            for n, fname in zip(self.local_length, self.local_stocks):
                n = max(0, n - self.stocks_size)
                if idx >= n:
                    idx -= n
                    continue

                self.df = pd.read_feather(fname)
                self.quick_remained = len(self.df) // (self.stocks_size * 4) - 1
                self.fname = fname
                break

        sample = self.df.iloc[idx:idx+self.stocks_size, 1:6].values.astype(np.float32)
        refdate = pd.to_datetime('1970-01-01')
        xdate = self.df.iloc[idx:idx+self.stocks_size, 0]
        days = (pd.to_datetime(xdate) - refdate).dt.days.values.astype(int)
        
        hx, lx = 0.9, -0.9
        hv, lv = 0.8, -1
        high = sample[:, 1].max()
        low = sample[:, 2].min()
        vmax = sample[:, 4].max()
        vmin = 0

        if high <= low or vmax <= 0:
            # invalid row
            samples = np.zeros_like(sample)
        else:
            sample[:,:4] = (sample[:,:4]-low) / (high-low) * (hx - lx) + lx
            sample[:,4] = (sample[:,4]-vmin) / (vmax-vmin) * (hv - lv) + lv
        
        sample = np.transpose(sample, [1, 0])
        return sample, days, self.fname
