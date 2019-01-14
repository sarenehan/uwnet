import numpy as np
import pandas as pd
from copy import deepcopy
import torch
from toolz import valmap
from torch.utils.data import Dataset
from itertools import product
import xarray as xr


def _stack_or_rename(x, **kwargs):
    for key, val in kwargs.items():
        if isinstance(val, str):
            x = x.rename({val: key})
        else:
            x = x.stack(**{key: val})
    return x


def _ds_slice_to_numpy_dict(ds):
    out = {}
    for key in ds.data_vars:
        out[key] = _to_numpy(ds[key])
    return out


def _to_numpy(x: xr.DataArray):
    dim_order = ['xbatch', 'xtime', 'xfeat']
    dims = [dim for dim in dim_order if dim in x.dims]
    return x.transpose(*dims).values


def _numpy_to_torch(x):
    y = torch.from_numpy(x).detach().float()
    return y.view(-1, 1, 1)


def _ds_slice_to_torch(ds):
    return valmap(_numpy_to_torch, _ds_slice_to_numpy_dict(ds))


class XRTimeSeries(Dataset):
    """A pytorch Dataset class for time series data in xarray format

    This function assumes the data has dimensions ['time', 'z', 'y', 'x'], and
    that the axes of the data arrays are all stored in that order.

    An individual "sample" is the full time time series from a single
    horizontal location. The time-varying variables in this sample will have
    shape (time, z, 1, 1).

    Examples
    --------
    >>> ds = xr.open_dataset("in.nc")
    >>> dataset = XRTimeSeries(ds)
    >>> dataset[0]

    """
    dims = ['time', 'z', 'x', 'y']

    def __init__(self, data, time_length=None):
        """
        Parameters
        ----------
        data : xr.DataArray
            An input dataset. This dataset must contain at least some variables
            with all of the dimensions ['time' , 'z', 'x', 'y'].
        time_length : int, optional
            The length of the time sequences to use, must evenly divide the
            total number of time points.
        """
        self.time_length = time_length or len(data.time)
        self.data = data
        self.data_vars = set(data.data_vars)
        self.dims = {key: data[key].dims for key in data.data_vars}
        self.constants = {
            key
            for key in data.data_vars
            if len({'x', 'y', 'time'} & set(data[key].dims)) == 0
        }
        self.setup_indices()

    def setup_indices(self):
        len_x = len(self.data['x'].values)
        len_y = len(self.data['y'].values)
        len_t = len(self.data['time'].values)

        x_iter = range(0, len_x, 1)
        y_iter = range(0, len_y, 1)
        t_iter = range(0, len_t, self.time_length)
        assert len_t % self.time_length == 0
        self.indices = list(product(t_iter, y_iter, x_iter))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t, y, x = self.indices[i]
        output_tensors = {}
        for key in self.data_vars:
            if key in self.constants:
                continue

            if 'z' in self.dims[key]:
                this_array_index = (slice(t, t + self.time_length),
                                    slice(None), y, x)
            else:
                this_array_index = (slice(t, t + self.time_length), None, y, x)

            sample = self.data[key].values[
                this_array_index][:, :, np.newaxis, np.newaxis]
            output_tensors[key] = sample.astype(np.float32)
        return output_tensors

    @property
    def time_dim(self):
        return self.dims[0][0]

    def torch_constants(self):
        return {
            key: torch.tensor(self.data[key].values, requires_grad=False)
            .float()
            for key in self.constants
        }

    @property
    def scale(self):
        std = self.std
        return valmap(lambda x: x.max(), std)


class ConditionalXRSampler(XRTimeSeries):
    """Same as XRTimeSeries, but only trained on a specific eta value.
    """

    def __init__(self, data, eta):
        """
        Parameters
        ----------
        data : xr.DataArray
            An input dataset. This dataset must contain at least some variables
            with all of the dimensions ['time' , 'z', 'x', 'y'].
        eta : int
            The value of eta to train on.
        """
        self.eta = eta
        super(ConditionalXRSampler, self).__init__(data, time_length=2)

    def setup_indices(self):
        indices = np.argwhere(
            self.data.eta.values == self.eta)
        self.indices = indices[indices[:, 0] < len(self.data.time) - 20]

    def get_two_steps_from_single_xy_batch(self, batch):
        batch_data = self.data.isel(batch)
        batch_next_time_step = deepcopy(batch)
        batch_next_time_step['time'] += 1
        batch_data_next_time_step = self.data.isel(batch)
        output_tensors = {}
        for key in self.data_vars:
            if key in self.constants:
                continue
            # torch.Size([64, 2, 34, 1, 1])
            if 'z' in self.dims[key]:
                output_start = batch_data[key].values[
                    :, :, np.newaxis, np.newaxis]
                output_stop = batch_data_next_time_step[key].values[
                    :, :, np.newaxis, np.newaxis]
                output_ = np.stack([output_start, output_stop], 1)
            else:
                output_start = batch_data[key].values[
                    :, np.newaxis, np.newaxis, np.newaxis]
                output_stop = batch_data_next_time_step[key].values[
                    :, np.newaxis, np.newaxis, np.newaxis]
                output_ = np.stack([output_start, output_stop], 1)
            output_tensors[key] = torch.FloatTensor(
                output_.astype(np.float32))
        return output_tensors

    def get_two_steps_from_mixed_xy_batch(self, batch):
        batch_data = self.data.isel(batch)
        batch_next_time_step = deepcopy(batch)
        batch_next_time_step['time'] += 1
        batch_data_next_time_step = self.data.isel(batch)
        output_tensors = {}
        for key in self.data_vars:
            if key in self.constants:
                continue
            # torch.Size([64, 2, 34, 1, 1])
            if 'z' in self.dims[key]:
                output_start = np.concatenate(
                    batch_data[key].values.T)[
                    :, :, np.newaxis, np.newaxis]
                output_stop = np.concatenate(
                    batch_data_next_time_step[key].values.T)[
                    :, :, np.newaxis, np.newaxis]
                output_ = np.stack([output_start, output_stop], 1)
            else:
                output_start = np.concatenate(
                    batch_data[key].values.T)[
                    :, np.newaxis, np.newaxis, np.newaxis]
                output_stop = np.concatenate(
                    batch_data_next_time_step[key].values.T)[
                    :, np.newaxis, np.newaxis, np.newaxis]
                output_ = np.stack([output_start, output_stop], 1)
            output_tensors[key] = torch.FloatTensor(
                output_.astype(np.float32))
        return output_tensors

    def get_two_time_steps_from_indices(self, indices):
        output_tensors = {}
        indices_next_step = indices.copy()
        indices_next_step[:, 0] += 1
        for key in self.data_vars:
            if key in self.constants:
                continue

            if 'z' in self.dims[key]:
                output_start = self.data[key].values[
                    indices[:, 0], :, indices[:, 1], indices[:, 2]
                ][:, :, np.newaxis, np.newaxis]
                output_stop = self.data[key].values[
                    indices_next_step[:, 0],
                    :,
                    indices_next_step[:, 1],
                    indices_next_step[:, 2]
                ][:, :, np.newaxis, np.newaxis]
                output_ = np.stack([output_start, output_stop], 1)
            else:
                output_start = self.data[key].values[
                    indices[:, 0], indices[:, 1], indices[:, 2]
                ][:, np.newaxis, np.newaxis, np.newaxis]
                output_stop = self.data[key].values[
                    indices_next_step[:, 0],
                    indices_next_step[:, 1],
                    indices_next_step[:, 2]
                ][:, np.newaxis, np.newaxis, np.newaxis]
                output_ = np.stack([output_start, output_stop], 1)
            output_tensors[key] = torch.FloatTensor(
                output_.astype(np.float32))
        return output_tensors


class TrainLoader(object):

    def __init__(self, train_data, batch_size):
        self.train_data = train_data
        self.batch_size = batch_size

        np.random.shuffle(self.train_data.indices)
        self.batches = np.array_split(
            self.train_data.indices,
            int(len(self.train_data) / batch_size)
        )
        # self.batches = self.get_batches(batch_size)

    def get_batches(self, batch_size):
        df = pd.DataFrame(self.train_data.indices, columns=['time', 'y', 'x'])
        batches = np.array([
            {'x': row.x, 'y': row.y, 'time': np.array(row.time)}
            for _, row in
            df.groupby(['y', 'x'], as_index=False).agg(
                lambda x: list(x)).iterrows()
        ])
        np.random.shuffle(batches)
        return batches

    def get_batches_old(self, batch_size):
        time_indices = np.unique(self.train_data.indices[:, 0])
        np.random.shuffle(time_indices)
        time_batches = np.array_split(
            time_indices, int(len(time_indices) / batch_size))
        batches = np.array([
            {'x': x, 'y': y, 'time': time_batch}
            for x in np.unique(self.train_data.indices[:, 2])
            for y in np.unique(self.train_data.indices[:, 1])
            for time_batch in time_batches
        ])
        np.random.shuffle(batches)
        return batches

    def get_batches_old_2(self, batch_size):
        time_indices = np.unique(self.train_data.indices[:, 0])
        y_indices = np.unique(self.train_data.indices[:, 1])
        x_indices = np.unique(self.train_data.indices[:, 2])
        np.random.shuffle(y_indices)
        np.random.shuffle(x_indices)

        x_batches = np.array_split(
            x_indices, round(len(x_indices) / (batch_size ** .5)))
        y_batches = np.array_split(
            y_indices, round(len(y_indices) / (batch_size ** .5)))

        # time_batches = np.array_split(
        #     time_indices, int(len(time_indices) / batch_size))
        batches = np.array([
            {'x': x_batch, 'y': y_batch, 'time': time_index}
            for x_batch in x_batches
            for y_batch in y_batches
            for time_index in time_indices
        ])
        np.random.shuffle(batches)
        return batches

    def __iter__(self):
        for batch_indices in self.batches:
            yield self.train_data.get_two_time_steps_from_indices(
                batch_indices)

    def __getitem__(self, i):
        return self.train_data.get_two_time_steps_from_indices(self.batches[i])

    def __len__(self):
        return len(self.batches)


def get_timestep(data):
    time_dim = 'time'
    time = data[time_dim]
    dt = np.diff(time)

    all_equal = dt.std() / dt.mean() < 1e-6
    if not all_equal:
        raise ValueError("Data must be uniformly sampled in time")

    if time.units.startswith('d'):
        return dt[0] * 86400
    elif time.units.startswith('s'):
        return dt[0]
    else:
        raise ValueError(
            f"Units of time are {time.units}, but must be either seconds"
            "or days")
