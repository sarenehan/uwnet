import torch


def numpy_3d_to_torch_flat(x):
    nz = x.shape[0]
    x = x.reshape((nz, -1)).T
    return torch.from_numpy(x).float()


def torch_flat_to_numpy_3d(x, shape):
    """Convert flat torch array to numpy and reshape

    Parameters
    ---------
    x
        (batch, feat) array
    shape : tuple
        The horizontal dimension sizes. Example: (ny, nx)
    """
    x = x.double().numpy()
    x = x.T
    nz = x.shape[0]
    orig_shape = (nz, ) + tuple(shape)
    return x.reshape(orig_shape).copy()


def step_model(step, dt, layer_mass, qt, sl, FQT, FSL, U, V, SST, SOLIN, **kw):
    """
    Take a step using the model

    Parameters
    ----------
    model
        Torch model with `step` function
    dt
        time step
    *args
        remaining inputs in kg -m -s  units.

    Returns
    -------
    outputs : dict of numpy arrays
        These outputs include LHF, SHF, RADTOA, RADSFC, Prec, qt, sl.
        All of these are in kg-m-s units as well.
    """

    s_to_day = 86400
    kg_kg_to_g_kg = 1000

    dt = dt / s_to_day

    # convert the units for the model
    x = {
        'qt': qt * kg_kg_to_g_kg,  # needs to be in g/kg
        'sl': sl,  # K
        'FQT': FQT * s_to_day * kg_kg_to_g_kg,
        'FSL': FSL * s_to_day,
        'U': U,
        'V': V,
        'SST': SST,
        'SOLIN': SOLIN,
    }

    nz, ny, nx = x['qt'].shape
    # call the neural network
    with torch.no_grad():
        # flatten numpy arrays and convert to torch
        x = {key: numpy_3d_to_torch_flat(val) for key, val in x.items()}

        # add layer mass
        x['layer_mass'] = torch.from_numpy(layer_mass).float()

        out, _ = step(x, dt)

    # flatten and scale outputs
    scales = {'qt': kg_kg_to_g_kg}
    out_np = {}
    for key, arr in out.items():
        arr = torch_flat_to_numpy_3d(arr, [ny, nx])
        if key in scales:
            arr /= scales[key]
        out_np[key] = arr
    return out_np


