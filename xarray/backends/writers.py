from __future__ import annotations

import importlib
import io
import os
from collections.abc import Callable, Hashable, Iterable, Mapping
from io import IOBase
from itertools import starmap
from numbers import Number
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal, get_args, overload

import numpy as np

from xarray import backends, conventions
from xarray.backends.api import (
    _normalize_path,
    delayed_close_after_writes,
)
from xarray.backends.common import (
    AbstractWritableDataStore,
    ArrayWriter,
    BytesIOProxy,
    _encode_variable_name,
)
from xarray.backends.locks import get_dask_scheduler
from xarray.backends.store import AbstractDataStore
from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree
from xarray.core.options import OPTIONS
from xarray.core.types import NetcdfWriteModes, ZarrWriteModes
from xarray.core.utils import emit_user_level_warning, is_remote_uri
from xarray.namedarray.pycompat import is_chunked_array

if TYPE_CHECKING:
    from dask.delayed import Delayed

    from xarray.backends import ZarrStore
    from xarray.backends.api import T_NetcdfEngine, T_NetcdfTypes
    from xarray.core.types import ZarrStoreLike


T_DataTreeNetcdfEngine = Literal["netcdf4", "h5netcdf", "pydap"]
T_DataTreeNetcdfTypes = Literal["NETCDF4"]


WRITEABLE_STORES: dict[T_NetcdfEngine, Callable] = {
    "netcdf4": backends.NetCDF4DataStore.open,
    "scipy": backends.ScipyDataStore,
    "h5netcdf": backends.H5NetCDFStore.open,
}


def get_writable_netcdf_store(
    target,
    engine: T_NetcdfEngine,
    *,
    format: T_NetcdfTypes | None,
    mode: NetcdfWriteModes,
    autoclose: bool,
    invalid_netcdf: bool,
    auto_complex: bool | None,
) -> AbstractWritableDataStore:
    """Create a store for writing to a netCDF file."""
    try:
        store_open = WRITEABLE_STORES[engine]
    except KeyError as err:
        raise ValueError(f"unrecognized engine for to_netcdf: {engine!r}") from err

    if format is not None:
        format = format.upper()  # type: ignore[assignment]

    kwargs = dict(autoclose=True) if autoclose else {}
    if invalid_netcdf:
        if engine == "h5netcdf":
            kwargs["invalid_netcdf"] = invalid_netcdf
        else:
            raise ValueError(
                f"unrecognized option 'invalid_netcdf' for engine {engine}"
            )
    if auto_complex is not None:
        kwargs["auto_complex"] = auto_complex

    return store_open(target, mode=mode, format=format, **kwargs)


def _validate_dataset_names(dataset: Dataset) -> None:
    """DataArray.name and Dataset keys must be a string or None"""

    def check_name(name: Hashable):
        if isinstance(name, str):
            if not name:
                raise ValueError(
                    f"Invalid name {name!r} for DataArray or Dataset key: "
                    "string must be length 1 or greater for "
                    "serialization to netCDF or zarr files"
                )
        elif name is not None:
            raise TypeError(
                f"Invalid name {name!r} for DataArray or Dataset key: "
                "must be either a string or None for serialization to netCDF "
                "or zarr files"
            )

    for k in dataset.variables:
        check_name(k)


def _validate_attrs(dataset, engine, invalid_netcdf=False):
    """`attrs` must have a string key and a value which is either: a number,
    a string, an ndarray, a list/tuple of numbers/strings, or a numpy.bool_.

    Notes
    -----
    A numpy.bool_ is only allowed when using the h5netcdf engine with
    `invalid_netcdf=True`.
    """

    valid_types = (str, Number, np.ndarray, np.number, list, tuple, bytes)
    if invalid_netcdf and engine == "h5netcdf":
        valid_types += (np.bool_,)

    def check_attr(name, value, valid_types):
        if isinstance(name, str):
            if not name:
                raise ValueError(
                    f"Invalid name for attr {name!r}: string must be "
                    "length 1 or greater for serialization to "
                    "netCDF files"
                )
        else:
            raise TypeError(
                f"Invalid name for attr: {name!r} must be a string for "
                "serialization to netCDF files"
            )

        if not isinstance(value, valid_types):
            raise TypeError(
                f"Invalid value for attr {name!r}: {value!r}. For serialization to "
                "netCDF files, its value must be of one of the following types: "
                f"{', '.join([vtype.__name__ for vtype in valid_types])}"
            )

        if isinstance(value, bytes) and engine == "h5netcdf":
            try:
                value.decode("utf-8")
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"Invalid value provided for attribute '{name!r}': {value!r}. "
                    "Only binary data derived from UTF-8 encoded strings is allowed "
                    f"for the '{engine}' engine. Consider using the 'netcdf4' engine."
                ) from e

            if b"\x00" in value:
                raise ValueError(
                    f"Invalid value provided for attribute '{name!r}': {value!r}. "
                    f"Null characters are not permitted for the '{engine}' engine. "
                    "Consider using the 'netcdf4' engine."
                )

    # Check attrs on the dataset itself
    for k, v in dataset.attrs.items():
        check_attr(k, v, valid_types)

    # Check attrs on each variable within the dataset
    for variable in dataset.variables.values():
        for k, v in variable.attrs.items():
            check_attr(k, v, valid_types)


def get_default_netcdf_write_engine(
    path_or_file: str | IOBase | None,
    format: T_NetcdfTypes | None,
) -> Literal["netcdf4", "h5netcdf", "scipy"]:
    """Return the default netCDF library to use for writing a netCDF file."""

    module_names = {
        "netcdf4": "netCDF4",
        "scipy": "scipy",
        "h5netcdf": "h5netcdf",
    }
    candidates = list(OPTIONS["netcdf_engine_order"])

    if format is not None:
        format = format.upper()  # type: ignore[assignment]
        if format not in {
            "NETCDF4",
            "NETCDF4_CLASSIC",
            "NETCDF3_64BIT",
            "NETCDF3_CLASSIC",
        }:
            raise ValueError(f"unexpected {format=}")
        # TODO: allow format='NETCDF4_CLASSIC' to default to using h5netcdf,
        # when the oldest supported version of h5netcdf supports it:
        # https://github.com/h5netcdf/h5netcdf/pull/283
        if format != "NETCDF4":
            candidates.remove("h5netcdf")
        if format not in {"NETCDF3_64BIT", "NETCDF3_CLASSIC"}:
            candidates.remove("scipy")

    nczarr_mode = isinstance(path_or_file, str) and path_or_file.endswith(
        "#mode=nczarr"
    )
    if nczarr_mode:
        candidates[:] = ["netcdf4"]

    if isinstance(path_or_file, IOBase):
        candidates.remove("netcdf4")

    for engine in candidates:
        module_name = module_names[engine]
        if importlib.util.find_spec(module_name) is not None:
            return engine

    if nczarr_mode:
        format_str = " in NCZarr format"
    else:
        format_str = f" with {format=}" if format is not None else ""
    libraries = ", ".join(module_names[c] for c in candidates)
    raise ValueError(
        f"cannot write NetCDF files{format_str} because none of the suitable "
        f"backend libraries ({libraries}) are installed"
    )


def _sanitize_unlimited_dims(dataset, unlimited_dims):
    msg_origin = "unlimited_dims-kwarg"
    if unlimited_dims is None:
        unlimited_dims = dataset.encoding.get("unlimited_dims", None)
        msg_origin = "dataset.encoding"
    if unlimited_dims is not None:
        if isinstance(unlimited_dims, str) or not isinstance(unlimited_dims, Iterable):
            unlimited_dims = [unlimited_dims]
        else:
            unlimited_dims = list(unlimited_dims)
        dataset_dims = set(dataset.dims)
        unlimited_dims = set(unlimited_dims)
        if undeclared_dims := (unlimited_dims - dataset_dims):
            msg = (
                f"Unlimited dimension(s) {undeclared_dims!r} declared in {msg_origin!r}, "
                f"but not part of current dataset dimensions. "
                f"Consider removing {undeclared_dims!r} from {msg_origin!r}."
            )
            if msg_origin == "unlimited_dims-kwarg":
                raise ValueError(msg)
            else:
                emit_user_level_warning(msg)
        return unlimited_dims


# multifile=True returns writer and datastore
@overload
def to_netcdf(
    dataset: Dataset,
    path_or_file: str | os.PathLike | None = None,
    mode: NetcdfWriteModes = "w",
    format: T_NetcdfTypes | None = None,
    group: str | None = None,
    engine: T_NetcdfEngine | None = None,
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    unlimited_dims: Iterable[Hashable] | None = None,
    append_dim: Hashable | None = None,
    compute: bool = True,
    *,
    multifile: Literal[True],
    invalid_netcdf: bool = False,
    auto_complex: bool | None = None,
) -> tuple[ArrayWriter, AbstractDataStore]: ...


# path=None writes to bytes or memoryview, depending on store
@overload
def to_netcdf(
    dataset: Dataset,
    path_or_file: None = None,
    mode: NetcdfWriteModes = "w",
    format: T_NetcdfTypes | None = None,
    group: str | None = None,
    engine: T_NetcdfEngine | None = None,
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    unlimited_dims: Iterable[Hashable] | None = None,
    append_dim: Hashable | None = None,
    compute: bool = True,
    multifile: Literal[False] = False,
    invalid_netcdf: bool = False,
    auto_complex: bool | None = None,
) -> memoryview: ...


# compute=False returns dask.Delayed
@overload
def to_netcdf(
    dataset: Dataset,
    path_or_file: str | os.PathLike,
    mode: NetcdfWriteModes = "w",
    format: T_NetcdfTypes | None = None,
    group: str | None = None,
    engine: T_NetcdfEngine | None = None,
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    unlimited_dims: Iterable[Hashable] | None = None,
    append_dim: Hashable | None = None,
    *,
    compute: Literal[False],
    multifile: Literal[False] = False,
    invalid_netcdf: bool = False,
    auto_complex: bool | None = None,
) -> Delayed: ...


# default return None
@overload
def to_netcdf(
    dataset: Dataset,
    path_or_file: str | os.PathLike | IOBase,
    mode: NetcdfWriteModes = "w",
    format: T_NetcdfTypes | None = None,
    group: str | None = None,
    engine: T_NetcdfEngine | None = None,
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    unlimited_dims: Iterable[Hashable] | None = None,
    append_dim: Hashable | None = None,
    compute: Literal[True] = True,
    multifile: Literal[False] = False,
    invalid_netcdf: bool = False,
    auto_complex: bool | None = None,
) -> None: ...


# if compute cannot be evaluated at type check time
# we may get back either Delayed or None
@overload
def to_netcdf(
    dataset: Dataset,
    path_or_file: str | os.PathLike,
    mode: NetcdfWriteModes = "w",
    format: T_NetcdfTypes | None = None,
    group: str | None = None,
    engine: T_NetcdfEngine | None = None,
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    unlimited_dims: Iterable[Hashable] | None = None,
    append_dim: Hashable | None = None,
    compute: bool = False,
    multifile: Literal[False] = False,
    invalid_netcdf: bool = False,
    auto_complex: bool | None = None,
) -> Delayed | None: ...


# if multifile cannot be evaluated at type check time
# we may get back either writer and datastore or Delayed or None
@overload
def to_netcdf(
    dataset: Dataset,
    path_or_file: str | os.PathLike,
    mode: NetcdfWriteModes = "w",
    format: T_NetcdfTypes | None = None,
    group: str | None = None,
    engine: T_NetcdfEngine | None = None,
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    unlimited_dims: Iterable[Hashable] | None = None,
    append_dim: Hashable | None = None,
    compute: bool = False,
    multifile: bool = False,
    invalid_netcdf: bool = False,
    auto_complex: bool | None = None,
) -> tuple[ArrayWriter, AbstractDataStore] | Delayed | None: ...


# Any
@overload
def to_netcdf(
    dataset: Dataset,
    path_or_file: str | os.PathLike | IOBase | None,
    mode: NetcdfWriteModes = "w",
    format: T_NetcdfTypes | None = None,
    group: str | None = None,
    engine: T_NetcdfEngine | None = None,
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    unlimited_dims: Iterable[Hashable] | None = None,
    append_dim: Hashable | None = None,
    compute: bool = False,
    multifile: bool = False,
    invalid_netcdf: bool = False,
    auto_complex: bool | None = None,
) -> tuple[ArrayWriter, AbstractDataStore] | memoryview | Delayed | None: ...


def to_netcdf(
    dataset: Dataset,
    path_or_file: str | os.PathLike | IOBase | None = None,
    mode: NetcdfWriteModes = "w",
    format: T_NetcdfTypes | None = None,
    group: str | None = None,
    engine: T_NetcdfEngine | None = None,
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    unlimited_dims: Iterable[Hashable] | None = None,
    append_dim: Hashable | None = None,
    compute: bool = True,
    multifile: bool = False,
    invalid_netcdf: bool = False,
    auto_complex: bool | None = None,
) -> tuple[ArrayWriter, AbstractDataStore] | memoryview | Delayed | None:
    """This function creates an appropriate datastore for writing a dataset to
    disk as a netCDF file

    See `Dataset.to_netcdf` for full API docs.

    The ``multifile`` argument is only for the private use of save_mfdataset.
    """
    if encoding is None:
        encoding = {}

    normalized_path = _normalize_path(path_or_file)

    if append_dim is not None:
        if mode != "a":
            raise ValueError("append_dim requires mode='a'")
        if multifile:
            raise NotImplementedError("append_dim is not supported with multifile=True")
        if normalized_path is None or isinstance(path_or_file, IOBase):
            raise ValueError("append_dim requires a path to an existing netCDF file")
        if is_remote_uri(normalized_path):
            raise NotImplementedError(
                "append_dim currently supports local netCDF files only"
            )
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(
                f"cannot append to non-existent netCDF file {normalized_path!r}"
            )

    if engine is None:
        engine = get_default_netcdf_write_engine(normalized_path, format)

    # validate Dataset keys, DataArray names, and attr keys/values
    _validate_dataset_names(dataset)
    _validate_attrs(dataset, engine, invalid_netcdf)
    # sanitize unlimited_dims
    unlimited_dims = _sanitize_unlimited_dims(dataset, unlimited_dims)

    autoclose = _get_netcdf_autoclose(dataset, engine)

    if normalized_path is None:
        if not compute:
            raise NotImplementedError(
                "to_netcdf() with compute=False is not yet implemented when "
                "returning a memoryview"
            )
        target = BytesIOProxy()
    else:
        target = normalized_path  # type: ignore[assignment]

    store = get_writable_netcdf_store(
        target,
        engine,
        mode=mode,
        format=format,
        autoclose=autoclose,
        invalid_netcdf=invalid_netcdf,
        auto_complex=auto_complex,
    )
    if group is not None:
        store = store.get_child_store(group)

    writer = ArrayWriter()

    # TODO: figure out how to refactor this logic (here and in save_mfdataset)
    # to avoid this mess of conditionals
    try:
        # TODO: allow this work (setting up the file for writing array data)
        # to be parallelized with dask
        dump_to_store(
            dataset,
            store,
            writer,
            encoding=encoding,
            unlimited_dims=unlimited_dims,
            append_dim=append_dim,
        )
        if autoclose:
            store.close()

        if multifile:
            return writer, store

        writes = writer.sync(compute=compute)

    finally:
        if not multifile and not autoclose:  # type: ignore[redundant-expr,unused-ignore]
            if compute:
                store.close()
            else:
                store.sync()

    if path_or_file is None:
        assert isinstance(target, BytesIOProxy)  # created in this function
        return target.getbuffer()

    if not compute:
        return delayed_close_after_writes(writes, store)

    return None


def _array_equal(left, right) -> bool:
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left, right))


def _attrs_compatible(provided, existing) -> bool:
    def normalize(value):
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value
        array = np.asarray(value)
        if array.size == 1:
            scalar = array.reshape(-1)[0]
            if isinstance(scalar, bytes):
                try:
                    return scalar.decode("utf-8")
                except UnicodeDecodeError:
                    return scalar
            return scalar
        return array

    for key, value in provided.items():
        if key not in existing:
            return False
        if not _array_equal(normalize(value), normalize(existing[key])):
            return False
    return True


def _existing_dimension_size(existing_variables, dim, stored_size) -> int:
    if stored_size is not None:
        return int(stored_size)
    sizes = {
        variable.sizes[dim]
        for variable in existing_variables.values()
        if dim in variable.dims
    }
    if len(sizes) > 1:
        raise ValueError(
            "existing netCDF variables disagree on the size of dimension "
            f"{dim!r}: {sizes}"
        )
    return next(iter(sizes), 0)


def _append_encoding(name, existing_variable):
    decoded = conventions.decode_cf_variable(name, existing_variable)
    encoding = decoded.encoding.copy()
    for key in ("source", "original_shape", "preferred_chunks"):
        encoding.pop(key, None)
    return decoded, encoding


def _validate_append_variable(name, new_variable, existing_variable, append_dim):
    if new_variable.dims != existing_variable.dims:
        if set(new_variable.dims) == set(existing_variable.dims):
            new_variable = new_variable.transpose(*existing_variable.dims)
        else:
            raise ValueError(
                f"variable {name!r} already exists with different dimensions: "
                f"{existing_variable.dims} != {new_variable.dims}"
            )

    for dim, size in existing_variable.sizes.items():
        if dim != append_dim and new_variable.sizes[dim] != size:
            raise ValueError(
                f"variable {name!r} already exists with a different size for "
                f"dimension {dim!r}: {size} != {new_variable.sizes[dim]}"
            )

    if not _attrs_compatible(new_variable.attrs, existing_variable.attrs):
        raise ValueError(
            f"variable {name!r} has incompatible attributes with the existing "
            "netCDF variable"
        )

    return new_variable


def _append_to_store(
    dataset,
    variables,
    attrs,
    store,
    writer,
    check_encoding,
    unlimited_dims,
    append_dim,
):
    if append_dim not in dataset.dims:
        raise ValueError(
            f"append_dim={append_dim!r} does not match any dataset dimension"
        )
    if dataset.sizes[append_dim] == 0:
        raise ValueError("cannot append zero elements to a netCDF dimension")
    if check_encoding:
        raise ValueError(
            "encoding cannot be supplied when append_dim is used; "
            "the existing netCDF variable encoding is reused"
        )
    if any(
        append_dim in variable.dims and is_chunked_array(variable._data)
        for variable in variables.values()
    ):
        raise NotImplementedError(
            "append_dim does not yet support chunked arrays"
        )

    existing_raw = store.get_variables()
    existing_dimensions = store.get_dimensions()
    existing_unlimited = set(store.get_encoding().get("unlimited_dims", ()))

    if append_dim not in existing_dimensions:
        raise ValueError(
            f"append_dim={append_dim!r} does not exist in the target netCDF file"
        )
    if append_dim not in existing_unlimited:
        raise ValueError(
            f"append_dim={append_dim!r} is not an unlimited dimension in the "
            "target netCDF file"
        )
    requested_unlimited = set(unlimited_dims or ())
    if not requested_unlimited <= existing_unlimited:
        raise ValueError(
            "append_dim cannot change which existing netCDF dimensions are unlimited"
        )

    variables = {_encode_variable_name(name): var for name, var in variables.items()}
    coordinate_names = {_encode_variable_name(name) for name in dataset.coords}

    new_names = set(variables) - set(existing_raw)
    if new_names:
        raise ValueError(
            "append_dim cannot add new variables to an existing netCDF file: "
            f"{new_names}"
        )

    existing = {}
    storage_encoding = {}
    for name, raw_variable in existing_raw.items():
        decoded, encoding = _append_encoding(name, raw_variable)
        existing[name] = decoded
        storage_encoding[name] = encoding

    existing_append_names = {
        name for name, variable in existing.items() if append_dim in variable.dims
    }
    new_append_names = {
        name for name, variable in variables.items() if append_dim in variable.dims
    }
    if missing := existing_append_names - new_append_names:
        raise ValueError(
            "all existing variables that use the append dimension must be present; "
            f"missing {missing}"
        )

    validated = {}
    for name, variable in variables.items():
        logical = _validate_append_variable(name, variable, existing[name], append_dim)
        logical = logical.copy(deep=False)
        logical.encoding = storage_encoding[name].copy()
        validated[name] = logical

    for name in coordinate_names:
        variable = validated.get(name)
        if variable is None or append_dim in variable.dims:
            continue
        existing_variable = existing[name]
        if not _array_equal(
            np.asarray(variable.data), np.asarray(existing_variable.data)
        ):
            raise ValueError(
                f"coordinate {name!r} differs from the target netCDF file"
            )

    encoded_variables, encoded_attrs = store.encode(validated, attrs)
    existing_attrs = store.get_attrs()
    if not _attrs_compatible(encoded_attrs, existing_attrs):
        raise ValueError(
            "dataset attributes are incompatible with the target netCDF file; "
            "append_dim does not modify existing metadata"
        )

    for name in new_append_names:
        raw_new = encoded_variables[name]
        raw_existing = existing_raw[name]
        if raw_new.dims != raw_existing.dims:
            raise ValueError(
                f"encoded dimensions for variable {name!r} do not match the target "
                "netCDF file"
            )
        if raw_new.dtype != raw_existing.dtype:
            raise ValueError(
                f"encoded dtype for variable {name!r} does not match the target "
                "netCDF file: "
                f"{raw_existing.dtype} != {raw_new.dtype}"
            )

    append_start = _existing_dimension_size(
        existing_raw, append_dim, existing_dimensions[append_dim]
    )
    append_stop = append_start + dataset.sizes[append_dim]

    resize_dimension = getattr(store, "resize_dimension", None)
    if resize_dimension is not None:
        resize_dimension(append_dim, append_stop)

    effective_unlimited_dims = existing_unlimited
    for name, variable in encoded_variables.items():
        if append_dim not in variable.dims:
            continue
        target, source = store.prepare_variable(
            name,
            variable,
            False,
            unlimited_dims=effective_unlimited_dims,
        )
        region = [slice(None)] * variable.ndim
        region[variable.dims.index(append_dim)] = slice(append_start, append_stop)
        writer.add(source, target, tuple(region))


def dump_to_store(
    dataset,
    store,
    writer=None,
    encoder=None,
    encoding=None,
    unlimited_dims=None,
    append_dim=None,
):
    """Store dataset contents to a backends.*DataStore object."""
    if writer is None:
        writer = ArrayWriter()

    if encoding is None:
        encoding = {}

    variables, attrs = conventions.encode_dataset_coordinates(dataset)

    check_encoding = set()
    for k, enc in encoding.items():
        variables[k].encoding = enc
        check_encoding.add(k)

    if encoder:
        variables, attrs = encoder(variables, attrs)

    if append_dim is None:
        store.store(
            variables, attrs, check_encoding, writer, unlimited_dims=unlimited_dims
        )
    else:
        _append_to_store(
            dataset,
            variables,
            attrs,
            store,
            writer,
            check_encoding,
            unlimited_dims,
            append_dim,
        )


def save_mfdataset(
    datasets,
    paths,
    mode="w",
    format=None,
    groups=None,
    engine=None,
    compute=True,
    **kwargs,
):
    """Write multiple datasets to disk as netCDF files simultaneously.

    This function is intended for use with datasets consisting of dask.array
    objects, in which case it can write the multiple datasets to disk
    simultaneously using a shared thread pool.

    When not using dask, it is no different than calling ``to_netcdf``
    repeatedly.

    Parameters
    ----------
    datasets : list of Dataset
        List of datasets to save.
    paths : list of str or list of path-like objects
        List of paths to which to save each corresponding dataset.
    mode : {"w", "a"}, optional
        Write ("w") or append ("a") mode. If mode="w", any existing file at
        these locations will be overwritten.
    format : {"NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", \
              "NETCDF3_CLASSIC"}, optional
        File format for the resulting netCDF file:

        * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API
          features.
        * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only
          netCDF 3 compatible API features.
        * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,
          which fully supports 2+ GB files, but is only compatible with
          clients linked against netCDF version 3.6.0 or later.
        * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not
          handle 2+ GB files very well.

        All formats are supported by the netCDF4-python library.
        scipy.io.netcdf only supports the last two formats.

        The default format is NETCDF4 if you are saving a file to disk and
        have the netCDF4-python library available. Otherwise, xarray falls
        back to using scipy to write netCDF files and defaults to the
        NETCDF3_64BIT format (scipy does not support netCDF4).
    groups : list of str, optional
        Paths to the netCDF4 group in each corresponding file to which to save
        datasets (only works for format="NETCDF4"). The groups will be created
        if necessary.
    engine : {"netcdf4", "h5netcdf", "scipy"}, optional
        Engine to use when writing netCDF files. If not provided, the
        default engine is chosen based on available dependencies, by default
        preferring "netcdf4" over "h5netcdf" over "scipy" (customizable via
        ``netcdf_engine_order`` in ``xarray.set_options()``).
    compute : bool
        If true compute immediately, otherwise return a
        ``dask.delayed.Delayed`` object that can be computed later.
    **kwargs : dict, optional
        Additional arguments are passed along to ``to_netcdf``.

    Examples
    --------
    Save a dataset into one netCDF per year of data:

    >>> ds = xr.Dataset(
    ...     {"a": ("time", np.linspace(0, 1, 48))},
    ...     coords={"time": pd.date_range("2010-01-01", freq="ME", periods=48)},
    ... )
    >>> ds
    <xarray.Dataset> Size: 768B
    Dimensions:  (time: 48)
    Coordinates:
      * time     (time) datetime64[us] 384B 2010-01-31 2010-02-28 ... 2013-12-31
    Data variables:
        a        (time) float64 384B 0.0 0.02128 0.04255 ... 0.9574 0.9787 1.0
    >>> years, datasets = zip(*ds.groupby("time.year"))
    >>> paths = [f"{y}.nc" for y in years]
    >>> xr.save_mfdataset(datasets, paths)
    """
    if mode == "w" and len(set(paths)) < len(paths):
        raise ValueError(
            "cannot use mode='w' when writing multiple datasets to the same path"
        )

    for obj in datasets:
        if not isinstance(obj, Dataset):
            raise TypeError(
                "save_mfdataset only supports writing Dataset "
                f"objects, received type {type(obj)}"
            )

    if groups is None:
        groups = [None] * len(datasets)

    if len({len(datasets), len(paths), len(groups)}) > 1:
        raise ValueError(
            "must supply lists of the same length for the "
            "datasets, paths and groups arguments to "
            "save_mfdataset"
        )

    writers, stores = zip(
        *[
            to_netcdf(
                ds,
                path,
                mode,
                format,
                group,
                engine,
                compute=compute,
                multifile=True,
                **kwargs,
            )
            for ds, path, group in zip(datasets, paths, groups, strict=True)
        ],
        strict=True,
    )

    try:
        writes = [w.sync(compute=compute) for w in writers]
    finally:
        for store in stores:
            if compute:
                store.close()
            else:
                store.sync()

    if not compute:
        import dask

        return dask.delayed(
            list(starmap(delayed_close_after_writes, zip(writes, stores, strict=True)))
        )


def get_writable_zarr_store(
    store: ZarrStoreLike | None = None,
    *,
    mode: ZarrWriteModes | None = None,
    group: str | None = None,
    consolidated: bool | None = None,
    append_dim: Hashable | None = None,
    region: Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None = None,
    safe_chunks: bool = True,
    align_chunks: bool = False,
    storage_options: dict[str, str] | None = None,
    zarr_format: int | None = None,
    write_empty_chunks: bool | None = None,
) -> backends.ZarrStore:
    """Create a store for writing to Zarr."""
    from xarray.backends.zarr import _choose_default_mode, _get_mappers

    kwargs, mapper = _get_mappers(storage_options=storage_options, store=store)
    mode = _choose_default_mode(mode=mode, append_dim=append_dim, region=region)

    if mode == "r+":
        already_consolidated = consolidated
        consolidate_on_close = False
    else:
        already_consolidated = False
        consolidate_on_close = consolidated or consolidated is None

    return backends.ZarrStore.open_group(
        store=mapper,
        mode=mode,
        group=group,
        consolidated=already_consolidated,
        consolidate_on_close=consolidate_on_close,
        append_dim=append_dim,
        write_region=region,
        safe_chunks=safe_chunks,
        align_chunks=align_chunks,
        zarr_format=zarr_format,
        write_empty=write_empty_chunks,
        **kwargs,
    )


# compute=True returns ZarrStore
@overload
def to_zarr(
    dataset: Dataset,
    store: ZarrStoreLike | None = None,
    mode: ZarrWriteModes | None = None,
    group: str | None = None,
    encoding: Mapping | None = None,
    *,
    compute: Literal[True] = True,
    consolidated: bool | None = None,
    append_dim: Hashable | None = None,
    region: Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None = None,
    safe_chunks: bool = True,
    align_chunks: bool = False,
    storage_options: dict[str, str] | None = None,
    write_empty_chunks: bool | None = None,
    chunkmanager_store_kwargs: dict[str, Any] | None = None,
) -> backends.ZarrStore: ...


# compute=False returns dask.Delayed
@overload
def to_zarr(
    dataset: Dataset,
    store: ZarrStoreLike | None = None,
    mode: ZarrWriteModes | None = None,
    group: str | None = None,
    encoding: Mapping | None = None,
    *,
    compute: Literal[False],
    consolidated: bool | None = None,
    append_dim: Hashable | None = None,
    region: Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None = None,
    safe_chunks: bool = True,
    align_chunks: bool = False,
    storage_options: dict[str, str] | None = None,
    write_empty_chunks: bool | None = None,
    chunkmanager_store_kwargs: dict[str, Any] | None = None,
) -> Delayed: ...


def to_zarr(
    dataset: Dataset,
    store: ZarrStoreLike | None = None,
    mode: ZarrWriteModes | None = None,
    group: str | None = None,
    encoding: Mapping | None = None,
    *,
    compute: bool = True,
    consolidated: bool | None = None,
    append_dim: Hashable | None = None,
    region: Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None = None,
    safe_chunks: bool = True,
    align_chunks: bool = False,
    storage_options: dict[str, str] | None = None,
    zarr_format: int | None = None,
    write_empty_chunks: bool | None = None,
    chunkmanager_store_kwargs: dict[str, Any] | None = None,
) -> backends.ZarrStore | Delayed:
    """This function creates an appropriate datastore for writing a dataset to
    a zarr ztore

    See `Dataset.to_zarr` for full API docs.
    """

    # validate Dataset keys, DataArray names
    _validate_dataset_names(dataset)

    # Load empty arrays to avoid bug saving zero length dimensions (Issue #5741)
    # TODO: delete when min dask>=2023.12.1
    # https://github.com/dask/dask/pull/10506
    for v in dataset.variables.values():
        if v.size == 0:
            v.load()

    if encoding is None:
        encoding = {}

    zstore = get_writable_zarr_store(
        store,
        mode=mode,
        group=group,
        consolidated=consolidated,
        append_dim=append_dim,
        region=region,
        safe_chunks=safe_chunks,
        align_chunks=align_chunks,
        storage_options=storage_options,
        zarr_format=zarr_format,
        write_empty_chunks=write_empty_chunks,
    )

    dataset = zstore._validate_and_autodetect_region(dataset)
    zstore._validate_encoding(encoding)

    writer = ArrayWriter()

    # TODO: figure out how to properly handle unlimited_dims
    try:
        dump_to_store(dataset, zstore, writer, encoding=encoding)
        writes = writer.sync(
            compute=compute, chunkmanager_store_kwargs=chunkmanager_store_kwargs
        )
    finally:
        if compute:
            zstore.close()

    if not compute:
        return delayed_close_after_writes(writes, zstore)

    return zstore


def _datatree_to_netcdf(
    dt: DataTree,
    filepath: str | PathLike | io.IOBase | None = None,
    mode: NetcdfWriteModes = "w",
    encoding: Mapping[str, Any] | None = None,
    unlimited_dims: Mapping | None = None,
    format: T_DataTreeNetcdfTypes | None = None,
    engine: T_DataTreeNetcdfEngine | None = None,
    group: str | None = None,
    write_inherited_coords: bool = False,
    compute: bool = True,
    invalid_netcdf: bool = False,
    auto_complex: bool | None = None,
) -> None | memoryview | Delayed:
    """Implementation of `DataTree.to_netcdf`."""

    if format not in [None, *get_args(T_DataTreeNetcdfTypes)]:
        raise ValueError("DataTree.to_netcdf only supports the NETCDF4 format")

    if engine not in [None, *get_args(T_DataTreeNetcdfEngine)]:
        raise ValueError(
            "DataTree.to_netcdf only supports the netcdf4 and h5netcdf engines"
        )

    normalized_path = _normalize_path(filepath)

    if engine is None:
        engine = get_default_netcdf_write_engine(
            path_or_file=normalized_path,
            format="NETCDF4",  # required for supporting groups
        )  # type: ignore[assignment]

    if group is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    if encoding is None:
        encoding = {}

    # In the future, we may want to expand this check to insure all the provided encoding
    # options are valid. For now, this simply checks that all provided encoding keys are
    # groups in the datatree.
    if set(encoding) - set(dt.groups):
        raise ValueError(
            f"unexpected encoding group name(s) provided: {set(encoding) - set(dt.groups)}"
        )

    if normalized_path is None:
        if not compute:
            raise NotImplementedError(
                "to_netcdf() with compute=False is not yet implemented when "
                "returning a memoryview"
            )
        target = BytesIOProxy()
    else:
        target = normalized_path  # type: ignore[assignment]

    if unlimited_dims is None:
        unlimited_dims = {}

    scheduler = get_dask_scheduler()
    have_chunks = any(
        v.chunks is not None for node in dt.subtree for v in node.variables.values()
    )
    autoclose = have_chunks and scheduler in ["distributed", "multiprocessing"]

    root_store = get_writable_netcdf_store(
        target,
        engine,  # type: ignore[arg-type]
        mode=mode,
        format=format,
        autoclose=autoclose,
        invalid_netcdf=invalid_netcdf,
        auto_complex=auto_complex,
    )

    writer = ArrayWriter()

    # TODO: allow this work (setting up the file for writing array data)
    # to be parallelized with dask
    try:
        for node in dt.subtree:
            at_root = node is dt
            dataset = node.to_dataset(inherit=write_inherited_coords or at_root)
            node_store = (
                root_store if at_root else root_store.get_child_store(node.path)
            )
            dump_to_store(
                dataset,
                node_store,
                writer,
                encoding=encoding.get(node.path),
                unlimited_dims=unlimited_dims.get(node.path),
            )

        if autoclose:
            root_store.close()

        writes = writer.sync(compute=compute)

    finally:
        if compute:
            root_store.close()
        else:
            root_store.sync()

    if filepath is None:
        assert isinstance(target, BytesIOProxy)  # created in this function
        return target.getbuffer()

    if not compute:
        return delayed_close_after_writes(writes, root_store)

    return None


def _datatree_to_zarr(
    dt: DataTree,
    store: ZarrStoreLike,
    mode: ZarrWriteModes = "w-",
    encoding: Mapping[str, Any] | None = None,
    group: str | None = None,
    write_inherited_coords: bool = False,
    *,
    compute: bool = True,
    consolidated: bool | None = None,
    append_dim: Hashable | None = None,
    region: Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None = None,
    safe_chunks: bool = True,
    align_chunks: bool = False,
    storage_options: dict[str, str] | None = None,
    zarr_format: int | None = None,
    write_empty_chunks: bool | None = None,
    chunkmanager_store_kwargs: dict[str, Any] | None = None,
) -> ZarrStore | Delayed:
    """Implementation of `DataTree.to_zarr`."""

    if group is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    if append_dim is not None:
        raise NotImplementedError(
            "specifying ``append_dim`` with ``DataTree.to_zarr`` has not been implemented"
        )

    if encoding is None:
        encoding = {}

    # In the future, we may want to expand this check to insure all the provided encoding
    # options are valid. For now, this simply checks that all provided encoding keys are
    # groups in the datatree.
    if set(encoding) - set(dt.groups):
        raise ValueError(
            f"unexpected encoding group name(s) provided: {set(encoding) - set(dt.groups)}"
        )

    root_store = get_writable_zarr_store(
        store,
        mode=mode,
        group=group,
        consolidated=consolidated,
        append_dim=append_dim,
        region=region,
        safe_chunks=safe_chunks,
        align_chunks=align_chunks,
        storage_options=storage_options,
        zarr_format=zarr_format,
        write_empty_chunks=write_empty_chunks,
    )

    writer = ArrayWriter()

    try:
        for rel_path, node in dt.subtree_with_keys:
            at_root = node is dt
            dataset = node.to_dataset(inherit=write_inherited_coords or at_root)
            # Use a relative path for group, because absolute paths are broken
            # with consolidated metadata in zarr 3.1.2 and earlier:
            # https://github.com/zarr-developers/zarr-python/pull/3428
            node_store = root_store if at_root else root_store.get_child_store(rel_path)

            dataset = node_store._validate_and_autodetect_region(dataset)
            node_store._validate_encoding(encoding)

            dump_to_store(
                dataset,
                node_store,
                writer,
                encoding=encoding.get(node.path),
            )
        writes = writer.sync(
            compute=compute, chunkmanager_store_kwargs=chunkmanager_store_kwargs
        )
    finally:
        if compute:
            root_store.close()

    if not compute:
        return delayed_close_after_writes(writes, root_store)

    return root_store


def _get_netcdf_autoclose(dataset: Dataset, engine: T_NetcdfEngine) -> bool:
    """Should we close files after each write operations?"""
    scheduler = get_dask_scheduler()
    have_chunks = any(v.chunks is not None for v in dataset.variables.values())

    autoclose = have_chunks and scheduler in ["distributed", "multiprocessing"]
    if autoclose and engine == "scipy":
        raise NotImplementedError(
            f"Writing netCDF files with the {engine} backend "
            f"is not currently supported with dask's {scheduler} scheduler"
        )
    return autoclose
