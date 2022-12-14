"""
Dataset utils for HSI classification
# Author: LQY @ HIT
# Modified by: HX @ HIT
# Email: 21B905003@stu.hit.edu.cn, 19B905007@stu.hit.edu.cn
"""
import numpy as np
import scipy.io as sio
from typing import Dict
from pathlib import Path
from numbers import Number
from collections import namedtuple

import torch

from datasets.hsi_dataset import HSIDataset, CachedHSIDataset


Datasets = namedtuple('Datasets', ['train', 'val', 'test'])


class HSIDatasetBuilder:
    """
    Base Dataset Builder for Hyper-spectral Image (HSI) Classification
    Args:
        root (str or pathlib.Path): Root directory of the meta hsi dataset files.
        filenames (Dict): Dict of the data filename and gt filename.
        train (float, optional): The proportion of training samples.
            notice: the quantity of training samples, not a decimal meant proportion.
        val (float, optional): The proportion of valid samples.
            notice: the quantity of valid samples, not a decimal meant proportion.
        seed (int, optional): Seed for splitting the dataset, which is independent of the seed
            set in the main progress and default to be 0.
        window_size (int, optional): The size of the square window for each sample.
            notice: should be a singular number.
        use_edge (bool, optional): Whether to use samples at edge whose window will exceed the
            image edge.
        dataset_name (str, optional): The name of the HSI dataset.
        num_classes (int, optional): The number of categories in the HSI dataset.
            If the arg is not given, it will be obtained by the builder. Although it is given,
            the builder will take a check.
        num_bands (int, optional): The number of spectral bands of the samples in the HSI dataset.
            Same to the num_samples, the builder can obtain and check.
    Usage (take IndianBuilder as an example):
        1. Init a builder:
            builder = IndianBuilder(seed=1)  # require kwarg
        2. Generate datasets:
        The builder support two building modes
        generator mode (HSIDataset) or cache mode (CachedHSIDataset).
            generator_datasets = builder.get_datasets()  # default
            cache_datasets = builder.get_datasets(cache_mode=True)  # require kwarg
            cache_train_set = cache_datasets.train
            cache_train_set = builder.get_train_dataset(cache_mode=True)
    """

    def __init__(
            self,
            root,
            filenames: Dict,
            train: Number = 200,
            val: Number = 50,
            seed: int = 0,
            window_size: int = 33,
            use_edge: bool = False,
            dataset_name: str = 'hsi',
            num_classes: int = None,
            num_bands: int = None,
    ):

        self.root = root if isinstance(root, Path) else Path(root)
        self.split_settings = {'train': train, 'val': val,
                               'window_size': window_size, 'use_edge': use_edge}
        self.seed = seed
        self.dataset_name = dataset_name
        self._num_classes = num_classes
        self._num_bands = num_bands
        self._cached = False

        self.meta_path = {}
        assert 'data' in filenames.keys() and 'gt' in filenames.keys()
        for k, filename in filenames.items():
            filename = Path(filename)
            assert '.mat' in filename.suffix, NotImplementedError(
                f'`{Path(filename).suffix}` is not supported yet, please load `.mat` file')
            self.meta_path[k] = self.root / filename

        if self.num_classes is None or self.num_bands is None:
            self.load_meta_dataset()

        dataset_name, seed = self.dataset_name, self.seed
        train, val = self.get_train_val()

        self.cache_path = {
            f'{dataset_type}_{file_type}':
                self.root / f'{dataset_name}' /
                f'{dataset_name}_t{train}_v{val}_{seed}_{dataset_type}_{file_type}.npy'
            for dataset_type in ('train', 'val', 'test') for file_type in ('data', 'gt')}

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_bands(self):
        return self._num_bands

    @property
    def cached(self):
        return self._cached

    def get_datasets(self, cache_mode: bool = False,
                     return_dict: bool = False, **kwargs):
        dataset_dict = {
            'train': self.get_train_dataset(cache_mode=cache_mode, **kwargs),
            'val': self.get_val_dataset(cache_mode=cache_mode, **kwargs),
            'test': self.get_test_dataset(cache_mode=cache_mode, **kwargs)}
        return dataset_dict if return_dict else Datasets(**dataset_dict)

    def load_splitted_dataset(self, dataset_type: str,
                              cache_mode: bool = False, **kwargs):
        if cache_mode:
            self.get_datasets_cache()
            data_path = self.cache_path[f'{dataset_type}_data']
            gt_path = self.cache_path[f'{dataset_type}_gt']
            print(f'Build {dataset_type} dataset of cache style.')
            return CachedHSIDataset(data_path, gt_path, **kwargs)
        else:
            data_path = self.meta_path['data']
            gt_path = self.meta_path['gt']
            info_path = self.get_split_info(**self.split_settings)
            print(f'Build {dataset_type} dataset of generator style.')
            return HSIDataset(data_path, gt_path, info_path, dataset_type, **kwargs)

    def get_split_info(self, gt=None, train=200, val=50, window_size=33, use_edge=False):
        info_path = self.root / f'{self.dataset_name}' / \
                    f'{self.dataset_name}_t{train}_v{val}_{self.seed}_split_info.mat'

        if not info_path.exists():
            gt = self.load_meta_dataset()[1] if gt is None else gt
            # Get proposal coords
            coords = self.get_proposal_coords(gt, window_size, use_edge)
            # Generate rand permutation for splitting dataset
            num_samples = len(coords)
            # TODO: The same num_cache cannot get the same RandPerm
            # RandPerm = np.random.permutation(num_samples)
            RandPerm = torch.randperm(
                num_samples, dtype=torch.int,
                generator=torch.Generator().manual_seed(self.seed),
            ).numpy()
            info_path.parent.mkdir(exist_ok=True)

            sio.savemat(info_path, {'coords': coords, 'RandPerm': RandPerm,
                                    'window_size': window_size, 'use_edge': int(use_edge),
                                    'train': train, 'val': val})
            print(f'Build split_info file at {info_path}.')

        return info_path

    def get_datasets_cache(self):
        """prepare dataset of cache mode"""
        if not self.cached:
            dataset_name, seed = self.dataset_name, self.seed
            is_cache_exist = {k: path.exists() for k, path in self.cache_path.items()}
            if all(is_cache_exist.values()):
                print(f'Loading {dataset_name} dataset from the seed={seed} cache data.')
            elif not any(is_cache_exist.values()):
                print(f'Splitting with seed={self.seed} for {dataset_name} dataset.')
                self.split_hsi_dataset(**self.split_settings)
            else:
                raise FileExistsError(
                    f'The existence status of an exception for data of seed={seed}: \n'
                    f'{is_cache_exist}')
        self._cached = True

    def split_hsi_dataset(self, train=200, val=50, window_size=33, use_edge=False):
        # Load meta dataset
        data, gt = self.load_meta_dataset()
        data = 1 * ((data - np.min(data)) / (np.max(data) - np.min(data)))
        data = data.transpose([2, 0, 1])  # H, W, C --> C, H, W

        info_path = self.get_split_info(gt, train, val, window_size, use_edge)

        splitted_datasets = self.split_with_info(data, gt, info_path)
        self.save_splitted_cache(splitted_datasets)

    @staticmethod
    def split_with_info(data, gt, info_path):
        C, H, W = data.shape  # Channel, Height, Width
        info = sio.loadmat(info_path)
        coords = info['coords']
        RandPerm = info['RandPerm'].squeeze()
        window_size = info['window_size'].item()
        train = info['train'].item()
        val = info['val'].item()
        coords_x, coords_y = coords[:, 0], coords[:, 1]

        # window size
        assert window_size % 2 == 1, ValueError(
            f"The window_size {window_size} ought to be a singular number")
        half_window = window_size // 2



        # Calculate numbers of samples in splitted subsets
        num_samples = len(coords)
        trainval = train + val
        if trainval < 1:
            train = int(num_samples * train)
            val = int(num_samples * val)
        elif train % 1 == 0 and val % 1 == 0:
            assert trainval >= 1
        test = num_samples - trainval
        print(f"total: {num_samples} | "
              f"train: {train}({100. * train / num_samples:.2}%) | "
              f"val {val}({100. * val / num_samples:.2}%) | "
              f"test {test}({100. * test / num_samples:.2}%)")

        gt = gt - 1
        output = {}
        output['train_data'] = np.zeros([train, C, window_size, window_size], dtype=np.float32)
        # output['train_spectral_data'] = np.zeros([train, C_spectral, half_window_spectral, half_window_spectral], dtype=np.float32)
        output['train_gt'] = np.zeros([train], dtype=np.int64)
        output['val_data'] = np.zeros([val, C, window_size, window_size], dtype=np.float32)
        # output['val_spectral_data'] = np.zeros([val, C_spectral, half_window_spectral, half_window_spectral], dtype=np.float32)
        output['val_gt'] = np.zeros([val], dtype=np.int64)
        output['test_data'] = np.zeros([test, C, window_size, window_size], dtype=np.float32)
        # output['test_spectral_data'] = np.zeros([test, C_spectral, half_window_spectral, half_window_spectral], dtype=np.float32)
        output['test_gt'] = np.zeros([test], dtype=np.int64)
        for i in range(train):
            output['train_data'][i, :, :, :] = \
                data[:,
                coords_x[RandPerm[i]] - half_window: coords_x[RandPerm[i]] + half_window + 1,
                coords_y[RandPerm[i]] - half_window: coords_y[RandPerm[i]] + half_window + 1]

            # output['train_spectral_data'][i, :, :, :] = \
            #     data[:,
            #     coords_x[RandPerm[i]] - half_window: coords_x[RandPerm[i]] + half_window + 1,
            #     coords_y[RandPerm[i]] - half_window: coords_y[RandPerm[i]] + half_window + 1]


            output['train_gt'][i] = \
                gt[coords_x[RandPerm[i]], coords_y[RandPerm[i]]].astype(np.int64)

        for i in range(val):
            output['val_data'][i, :, :, :] = \
                data[:,
                coords_x[RandPerm[i + train]] - half_window: coords_x[RandPerm[i + train]] + half_window + 1,
                coords_y[RandPerm[i + train]] - half_window: coords_y[RandPerm[i + train]] + half_window + 1]

            # output['val_spectral_data'][i, :, :, :] = \
            #     data[:,
            #     coords_x[RandPerm[i + train]] - half_window: coords_x[RandPerm[i + train]] + half_window + 1,
            #     coords_y[RandPerm[i + train]] - half_window: coords_y[RandPerm[i + train]] + half_window + 1]

            output['val_gt'][i] = \
                gt[coords_x[RandPerm[i + train]], coords_y[RandPerm[i + train]]].astype(np.int64)

        for i in range(test):
            output['test_data'][i, :, :, :] = \
                data[:,
                coords_x[RandPerm[i + trainval]] - half_window: coords_x[RandPerm[i + trainval]] + half_window + 1,
                coords_y[RandPerm[i + trainval]] - half_window: coords_y[RandPerm[i + trainval]] + half_window + 1]

            # output['test_spectral_data'][i, :, :, :] = \
            #     data[:,
            #     coords_x[RandPerm[i + trainval]] - half_window: coords_x[RandPerm[i + trainval]] + half_window + 1,
            #     coords_y[RandPerm[i + trainval]] - half_window: coords_y[RandPerm[i + trainval]] + half_window + 1]

            output['test_gt'][i] = \
                gt[coords_x[RandPerm[i + trainval]], coords_y[RandPerm[i + trainval]]].astype(np.int64)

        print("Splitting successfully.")

        return output

    def save_splitted_cache(self, output):
        for dataset_type in ('train', 'val', 'test'):
            for file_type in ('data', 'gt'):
                self.cache_path[f'{dataset_type}_{file_type}'].parent.mkdir(exist_ok=True)
                np.save(self.cache_path[f'{dataset_type}_{file_type}'],
                        output[f'{dataset_type}_{file_type}'])
        print("Data cache saved.")

    def get_proposal_coords(self, gt, window_size=33, use_edge=False):
        H, W = gt.shape

        # window size
        assert window_size % 2 == 1, ValueError(
            f"The window_size {window_size} ought to be a singular number")
        half_window = window_size // 2

        # Edge processing
        # The edge samples are referred negative samples which are not used.
        if use_edge:
            raise NotImplementedError
        else:
            mask = np.zeros([H, W])  # 0:reject  1:accept
            mask[half_window + 1: -1 - half_window + 1, half_window + 1: -1 - half_window + 1] = 1
            gt = gt * mask
        coords_x, coords_y = np.nonzero(gt)
        coords = np.stack([coords_x, coords_y], -1)

        return coords

    def load_meta_dataset(self):
        data = {k: v for k, v in sio.loadmat(self.meta_path['data']).items()
                if isinstance(v, np.ndarray)}
        gt = {k: v for k, v in sio.loadmat(self.meta_path['gt']).items()
              if isinstance(v, np.ndarray) and 'map' not in k}
        assert len(data) == 1 and len(gt) == 1, ValueError('Description Reading the MAT file conflicts.')
        data, gt = list(data.values())[0], list(gt.values())[0]
        _, _, num_bands = data.shape  # Channel, Height, Width
        if self.num_bands is not None:
            assert self.num_bands == num_bands
        else:
            self._num_bands = num_bands
        num_classes = int(np.max(gt))
        if self.num_classes is not None:
            assert self.num_classes == num_classes
        else:
            self._num_classes = num_classes
        return data, gt

    def get_train_dataset(self, **kwargs):
        return self.load_splitted_dataset('train', **kwargs)

    def get_val_dataset(self, **kwargs):
        return self.load_splitted_dataset('val', **kwargs)

    def get_test_dataset(self, **kwargs):
        return self.load_splitted_dataset('test', **kwargs)

    def get_num_classes(self):
        return self.num_classes

    def get_num_bands(self):
        return self.num_bands

    def get_train_val(self):
        s = self.split_settings
        return s['train'], s['val']

    def __repr__(self) -> str:
        _repr_indent = 4
        head = self.__class__.__name__
        body = [f"Generating seed: {self.seed} ",
                f"Split settings: {self.split_settings.items()} ",
                f"Root location: {self.root} ",
                f"Dataset info: named {self.dataset_name}, "
                f"{self.num_bands} bands, "
                f"{self.num_classes} categories"]
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> str:
        return ""


class IndianBuilder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=50,
                 seed: int=None, window_size: int=33,
                 use_edge: bool=False,):
        filenames = {'data': 'Indian_pines_corrected.mat',
                     'gt': 'Indian_pines_gt.mat'}
        dataset_name = 'Indian_pines'
        num_classes = 16
        num_bands = 200
        super(IndianBuilder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands)


class PaviaBuilder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=50,
                 seed: int=None, window_size: int=33,
                 use_edge: bool=False,):
        filenames = {'data': 'Pavia.mat',
                     'gt': 'Pavia_groundtruth.mat'}
        dataset_name = 'Pavia'
        num_classes = 9
        num_bands = 103
        super(PaviaBuilder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands)


class KSCBuilder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=150, val: Number=50,
                 seed: int=None, window_size: int=33,
                 use_edge: bool=False,):
        filenames = {'data': 'Kennedy_denoise.mat',
                     'gt': 'KSC_gt.mat'}
        dataset_name = 'KSC'
        num_classes = 13
        num_bands = 176
        super(KSCBuilder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands)


class SalinasBuilder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=50,
                 seed: int=None, window_size: int=33,
                 use_edge: bool=False,):
        filenames = {'data': 'Salinas_corrected.mat',
                     'gt': 'Salinas_gt.mat'}
        dataset_name = 'Salinas'
        num_classes = 16
        num_bands = 204
        super(SalinasBuilder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands)


class CASIBuilder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=50,
                 seed: int=None, window_size: int=33,
                 use_edge: bool=False,):
        filenames = {'data': 'CASI.mat',
                     'gt': 'CASI_gnd_flag.mat'}
        dataset_name = 'CASI'
        num_classes = 15
        num_bands = 144
        super(CASIBuilder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands)


if __name__ == '__main__':
    # an inspection of the implementation
    root = Path(r'G:/Transfer所有相关Trans/Indian')
    root = root if root.exists() else input('Dataset path of hsi dataset:')
    Builders = (IndianBuilder, KSCBuilder,
                # PaviaBuilder, SalinasBuilder, CASIBuilder
                )
    import time
    from tqdm import tqdm
    for Builder in Builders:
        _time = time.time()
        builder = Builder(root, seed=1)
        print(f'builder init time: {time.time() - _time}')
        _time = time.time()
        print(builder, end='\n' * 3)
        datasets_0 = builder.get_datasets(cache_mode=False)
        print(f'datasets_0 init time: {time.time() - _time}')
        _time = time.time()
        print(datasets_0, end='\n' * 3)
        datasets_1 = builder.get_datasets(cache_mode=True)
        print(f'datasets_1 init time: {time.time() - _time}')
        _time = time.time()
        print(datasets_1, end='\n' * 3)
        # datasets_0 = [builder.get_train_dataset(cache_mode=False)]
        # datasets_1 = [builder.get_train_dataset(cache_mode=True)]
        assert len(datasets_0) == len(datasets_1)
        for i in range(len(datasets_0)):
            assert len(datasets_0[i]) == len(datasets_1[i])
            for j in tqdm(range(len(datasets_0[i]))):
                sample_0, target_0 = datasets_0[i][j]
                sample_1, target_1 = datasets_1[i][j]
                assert torch.all(torch.eq(sample_0, sample_1))
                assert torch.all(torch.eq(target_0, target_1))
        print(f'The {builder.__class__} passed the inspection.')

