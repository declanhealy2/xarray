from __future__ import annotations

import warnings

import numpy as np

from xarray.core import dtypes, duck_array_ops, nputils, utils
from xarray.core.duck_array_ops import (
    astype,
    count,
    fillna,
    isnull,
    sum_where,
    where,
    where_method,
)


def _maybe_null_out(result, axis, mask, min_count=1):
    """
    xarray version of pandas.core.nanops._maybe_null_out
    """
    if axis is not None and getattr(result, "ndim", False):
        null_mask = (
            np.take(mask.shape, axis).prod()
            - duck_array_ops.sum(mask, axis)
            - min_count
        ) < 0
        dtype, fill_value = dtypes.maybe_promote(result.dtype)
        result = where(null_mask, fill_value, astype(result, dtype))

    elif (dtype := getattr(result, "dtype", None)) and getattr(
        dtype, "kind", None
    ) not in {"m", "M"}:
        null_mask = mask.size - duck_array_ops.sum(mask)
        result = where(null_mask < min_count, np.nan, result)

    return result


def _nan_argminmax_object(func, fill_value, value, axis=None, **kwargs):
    """In house nanargmin, nanargmax for object arrays. Always return integer
    type
    """
    valid_count = count(value, axis=axis)
    value = fillna(value, fill_value)
    data = getattr(np, func)(value, axis=axis, **kwargs)

    # TODO This will evaluate dask arrays and might be costly.
    if duck_array_ops.array_any(valid_count == 0):
        raise ValueError("All-NaN slice encountered")

    return data


def _nan_minmax_object(func, fill_value, value, axis=None, **kwargs):
    """In house nanmin and nanmax for object array"""
    valid_count = count(value, axis=axis)
    filled_value = fillna(value, fill_value)
    data = getattr(np, func)(filled_value, axis=axis, **kwargs)
    if not hasattr(data, "dtype"):  # scalar case
        data = fill_value if valid_count == 0 else data
        # we've computed a single min, max value of type object.
        # don't let np.array turn a tuple back into an array
        return utils.to_0d_object_array(data)
    return where_method(data, valid_count != 0)


def _has_native_nanop(a, name):
    xp = duck_array_ops.get_array_namespace(a)
    return getattr(xp, name, None) is not None


def _nan_count(mask, xp, axis=None, keepdims=False):
    return xp.sum(~mask, axis=axis, keepdims=keepdims)


def _nan_result(value, count, xp, minimum_count=1):
    fill_value = xp.asarray(np.nan, dtype=value.dtype)
    return xp.where(count < minimum_count, fill_value, value)


def _nanmean_array_api(a, axis=None, dtype=None, keepdims=False):
    xp = duck_array_ops.get_array_namespace(a)
    mask = isnull(a)
    value = astype(a, dtype) if dtype is not None else a
    value = xp.where(mask, xp.asarray(0, dtype=value.dtype), value)
    count = _nan_count(mask, xp, axis=axis, keepdims=keepdims)
    total = xp.sum(value, axis=axis, keepdims=keepdims)
    return _nan_result(total / count, count, xp)


def _nanvar_array_api(a, axis=None, dtype=None, ddof=0):
    xp = duck_array_ops.get_array_namespace(a)
    mask = isnull(a)
    value = astype(a, dtype) if dtype is not None else a
    count = _nan_count(mask, xp, axis=axis)
    mean = _nanmean_array_api(value, axis=axis, keepdims=True)
    squared = xp.where(
        mask, xp.asarray(0, dtype=value.dtype), xp.abs(value - mean) ** 2
    )
    denominator = xp.where(count > ddof, count - ddof, 1)
    variance = xp.sum(squared, axis=axis) / denominator
    return _nan_result(variance, count, xp, minimum_count=ddof + 1)


def _nanminmax_array_api(a, name, fill_value, axis=None):
    xp = duck_array_ops.get_array_namespace(a)
    mask = isnull(a)
    value = xp.where(mask, xp.asarray(fill_value, dtype=a.dtype), a)
    result = getattr(xp, name)(value, axis=axis)
    return _nan_result(result, _nan_count(mask, xp, axis=axis), xp)


def nanmin(a, axis=None, out=None):
    if dtypes.is_object(a.dtype):
        return _nan_minmax_object("min", dtypes.get_pos_infinity(a.dtype), a, axis)

    if not _has_native_nanop(a, "nanmin"):
        return _nanminmax_array_api(a, "min", np.inf, axis=axis)

    return nputils.nanmin(a, axis=axis)


def nanmax(a, axis=None, out=None):
    if dtypes.is_object(a.dtype):
        return _nan_minmax_object("max", dtypes.get_neg_infinity(a.dtype), a, axis)

    if not _has_native_nanop(a, "nanmax"):
        return _nanminmax_array_api(a, "max", -np.inf, axis=axis)

    return nputils.nanmax(a, axis=axis)


def nanargmin(a, axis=None):
    if dtypes.is_object(a.dtype):
        fill_value = dtypes.get_pos_infinity(a.dtype)
        return _nan_argminmax_object("argmin", fill_value, a, axis=axis)

    return nputils.nanargmin(a, axis=axis)


def nanargmax(a, axis=None):
    if dtypes.is_object(a.dtype):
        fill_value = dtypes.get_neg_infinity(a.dtype)
        return _nan_argminmax_object("argmax", fill_value, a, axis=axis)

    return nputils.nanargmax(a, axis=axis)


def nansum(a, axis=None, dtype=None, out=None, min_count=None):
    mask = isnull(a)
    result = sum_where(a, axis=axis, dtype=dtype, where=mask)
    if min_count is not None:
        return _maybe_null_out(result, axis, mask, min_count)
    else:
        return result


def _nanmean_ddof_object(ddof, value, axis=None, dtype=None, **kwargs):
    """In house nanmean. ddof argument will be used in _nanvar method"""
    valid_count = count(value, axis=axis)
    value = fillna(value, 0)
    # As dtype inference is impossible for object dtype, we assume float
    # https://github.com/dask/dask/issues/3162
    if dtype is None and value.dtype.kind == "O":
        dtype = float

    data = np.sum(value, axis=axis, dtype=dtype, **kwargs)
    data = data / (valid_count - ddof)
    return where_method(data, valid_count != 0)


def nanmean(a, axis=None, dtype=None, out=None):
    if dtypes.is_object(a.dtype):
        return _nanmean_ddof_object(0, a, axis=axis, dtype=dtype)

    if not _has_native_nanop(a, "nanmean"):
        return _nanmean_array_api(a, axis=axis, dtype=dtype)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", r"Mean of empty slice", category=RuntimeWarning
        )

        return nputils.nanmean(a, axis=axis, dtype=dtype)


def nanmedian(a, axis=None, out=None):
    # The dask algorithm works by rechunking to one chunk along axis
    # Make sure we trigger the dask error when passing all dimensions
    # so that we don't rechunk the entire array to one chunk and
    # possibly blow memory
    if axis is not None and len(np.atleast_1d(axis)) == a.ndim:
        axis = None
    return nputils.nanmedian(a, axis=axis)


def _nanvar_object(value, axis=None, ddof=0, keepdims=False, **kwargs):
    value_mean = _nanmean_ddof_object(
        ddof=0, value=value, axis=axis, keepdims=True, **kwargs
    )
    squared = (astype(value, value_mean.dtype) - value_mean) ** 2
    return _nanmean_ddof_object(ddof, squared, axis=axis, keepdims=keepdims, **kwargs)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0):
    if dtypes.is_object(a.dtype):
        return _nanvar_object(a, axis=axis, dtype=dtype, ddof=ddof)

    if not _has_native_nanop(a, "nanvar"):
        return _nanvar_array_api(a, axis=axis, dtype=dtype, ddof=ddof)

    return nputils.nanvar(a, axis=axis, dtype=dtype, ddof=ddof)


def nanstd(a, axis=None, dtype=None, out=None, ddof=0):
    if not _has_native_nanop(a, "nanstd"):
        xp = duck_array_ops.get_array_namespace(a)
        return xp.sqrt(_nanvar_array_api(a, axis=axis, dtype=dtype, ddof=ddof))

    return nputils.nanstd(a, axis=axis, dtype=dtype, ddof=ddof)


def nanprod(a, axis=None, dtype=None, out=None, min_count=None):
    if not _has_native_nanop(a, "nanprod"):
        xp = duck_array_ops.get_array_namespace(a)
        mask = isnull(a)
        value = astype(a, dtype) if dtype is not None else a
        result = xp.prod(
            xp.where(mask, xp.asarray(1, dtype=value.dtype), value), axis=axis
        )
        if min_count is not None:
            return _nan_result(result, _nan_count(mask, xp, axis=axis), xp, min_count)
        return result

    mask = isnull(a)
    result = nputils.nanprod(a, axis=axis, dtype=dtype)
    if min_count is not None:
        return _maybe_null_out(result, axis, mask, min_count)
    else:
        return result


def nancumsum(a, axis=None, dtype=None, out=None):
    if not _has_native_nanop(a, "nancumsum"):
        xp = duck_array_ops.get_array_namespace(a)
        value = astype(a, dtype) if dtype is not None else a
        return xp.cumsum(
            xp.where(isnull(value), xp.asarray(0, dtype=value.dtype), value), axis=axis
        )

    return nputils.nancumsum(a, axis=axis, dtype=dtype)


def nancumprod(a, axis=None, dtype=None, out=None):
    if not _has_native_nanop(a, "nancumprod"):
        xp = duck_array_ops.get_array_namespace(a)
        value = astype(a, dtype) if dtype is not None else a
        return xp.cumprod(
            xp.where(isnull(value), xp.asarray(1, dtype=value.dtype), value), axis=axis
        )

    return nputils.nancumprod(a, axis=axis, dtype=dtype)
