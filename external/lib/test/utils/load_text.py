import numpy as np
import pandas as pd


def load_text_numpy(path, delimiter, dtype):
    if not isinstance(delimiter, (tuple, list)):
        return np.loadtxt(path, delimiter=delimiter, dtype=dtype)
    for d in delimiter:
        try:
            return np.loadtxt(path, delimiter=d, dtype=dtype)
        except:
            pass

    raise Exception(f'Could not read file {path}')


def load_text_pandas(path, delimiter, dtype):
    if not isinstance(delimiter, (tuple, list)):
        return pd.read_csv(
            path,
            delimiter=delimiter,
            header=None,
            dtype=dtype,
            na_filter=False,
            low_memory=False,
        ).values

    for d in delimiter:
        try:
            return pd.read_csv(
                path,
                delimiter=d,
                header=None,
                dtype=dtype,
                na_filter=False,
                low_memory=False,
            ).values

        except Exception as e:
            pass

    raise Exception(f'Could not read file {path}')


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)