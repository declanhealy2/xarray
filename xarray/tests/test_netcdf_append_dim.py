from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import xarray as xr


def _available_engines() -> list[str]:
    engines = ["scipy"]
    if importlib.util.find_spec("netCDF4") is not None:
        engines.append("netcdf4")
    if importlib.util.find_spec("h5netcdf") is not None:
        engines.append("h5netcdf")
    return engines


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_mlx_array(tmp_path, engine):
    mx = pytest.importorskip("mlx.core")
    path = tmp_path / "mlx.nc"
    first = xr.Dataset(
        {"signal": ("cpi", mx.array([1.0, 2.0], dtype=mx.float32))},
        coords={"cpi": [0, 1]},
    )
    first.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])
    second = xr.Dataset(
        {"signal": ("cpi", mx.array([3.0, 4.0], dtype=mx.float32))},
        coords={"cpi": [2, 3]},
    )
    second.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")

    with xr.open_dataset(path, engine=engine) as actual:
        np.testing.assert_array_equal(actual["signal"], [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_extends_unlimited_dimension(tmp_path, engine):
    path = tmp_path / "append.nc"
    first = xr.Dataset(
        {"signal": (("cpi", "delay"), np.arange(6, dtype=np.float32).reshape(2, 3))},
        coords={"cpi": np.array([0, 1], dtype=np.int32), "delay": [10, 20, 30]},
        attrs={"kind": "stream"},
    )
    first["signal"].attrs["units"] = "V"
    first.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])

    second = xr.Dataset(
        {
            "signal": (
                ("cpi", "delay"),
                np.arange(6, 12, dtype=np.float32).reshape(2, 3),
            )
        },
        coords={"cpi": np.array([2, 3], dtype=np.int32), "delay": [10, 20, 30]},
        attrs={"kind": "stream"},
    )
    second["signal"].attrs["units"] = "V"
    second.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")

    with xr.open_dataset(path, engine=engine) as actual:
        actual.load()
        assert actual.sizes == {"cpi": 4, "delay": 3}
        np.testing.assert_array_equal(actual["cpi"], [0, 1, 2, 3])
        np.testing.assert_array_equal(
            actual["signal"], np.arange(12, dtype=np.float32).reshape(4, 3)
        )


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_repeated_multi_record_append(tmp_path, engine):
    path = tmp_path / "append.nc"
    xr.Dataset({"x": ("cpi", np.array([0, 1], dtype=np.int32))}).to_netcdf(
        path, engine=engine, unlimited_dims=["cpi"]
    )

    for start in (2, 5):
        values = np.arange(start, start + 3, dtype=np.int32)
        xr.Dataset({"x": ("cpi", values)}).to_netcdf(
            path, engine=engine, mode="a", append_dim="cpi"
        )

    with xr.open_dataset(path, engine=engine) as actual:
        np.testing.assert_array_equal(actual["x"], np.arange(8, dtype=np.int32))


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_reuses_datetime_encoding(tmp_path, engine):
    path = tmp_path / "append-time.nc"
    first = xr.Dataset(
        {"x": ("time", np.array([1.0, 2.0]))},
        coords={"time": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]")},
    )
    first.to_netcdf(path, engine=engine, unlimited_dims=["time"])

    second = xr.Dataset(
        {"x": ("time", np.array([3.0, 4.0]))},
        coords={"time": np.array(["2026-02-01", "2026-02-02"], dtype="datetime64[ns]")},
    )
    second.to_netcdf(path, engine=engine, mode="a", append_dim="time")

    expected = np.array(
        ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"],
        dtype="datetime64[ns]",
    )
    with xr.open_dataset(path, engine=engine) as actual:
        np.testing.assert_array_equal(actual["time"], expected)
        np.testing.assert_array_equal(actual["x"], [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_reuses_packed_encoding(tmp_path, engine):
    path = tmp_path / "append-packed.nc"
    first = xr.Dataset(
        {"x": ("cpi", np.array([1.0, 2.0]))},
        coords={"cpi": np.array([0, 1], dtype=np.int32)},
    )
    first.to_netcdf(
        path,
        engine=engine,
        unlimited_dims=["cpi"],
        encoding={"x": {"dtype": "int16", "scale_factor": 0.1, "_FillValue": -9999}},
    )

    second = xr.Dataset(
        {"x": ("cpi", np.array([3.0, 4.0]))},
        coords={"cpi": np.array([2, 3], dtype=np.int32)},
    )
    second.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")

    with xr.open_dataset(path, engine=engine) as actual:
        np.testing.assert_allclose(actual["x"], [1.0, 2.0, 3.0, 4.0])
        assert np.dtype(actual["x"].encoding["dtype"]) == np.dtype("int16")
        assert actual["x"].encoding["scale_factor"] == pytest.approx(0.1)


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_dataarray(tmp_path, engine):
    path = tmp_path / "append-array.nc"
    first = xr.DataArray(
        np.array([1.0, 2.0]), dims="cpi", coords={"cpi": [0, 1]}, name="x"
    )
    first.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])

    second = xr.DataArray(
        np.array([3.0, 4.0]), dims="cpi", coords={"cpi": [2, 3]}, name="x"
    )
    second.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")

    with xr.open_dataarray(path, engine=engine) as actual:
        np.testing.assert_array_equal(actual, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(actual["cpi"], [0, 1, 2, 3])


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_requires_append_mode(tmp_path, engine):
    path = tmp_path / "append.nc"
    ds = xr.Dataset({"x": ("cpi", [1])})
    ds.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])

    with pytest.raises(ValueError, match="append_dim requires mode='a'"):
        ds.to_netcdf(path, engine=engine, append_dim="cpi")


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_requires_existing_unlimited_dimension(tmp_path, engine):
    path = tmp_path / "fixed.nc"
    ds = xr.Dataset({"x": ("cpi", [1, 2])})
    ds.to_netcdf(path, engine=engine)

    with pytest.raises(ValueError, match="is not an unlimited dimension"):
        ds.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_rejects_changed_fixed_coordinate(tmp_path, engine):
    path = tmp_path / "coords.nc"
    first = xr.Dataset(
        {"x": (("cpi", "delay"), np.ones((1, 2)))},
        coords={"cpi": [0], "delay": [10, 20]},
    )
    first.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])

    second = xr.Dataset(
        {"x": (("cpi", "delay"), np.ones((1, 2)))},
        coords={"cpi": [1], "delay": [10, 21]},
    )
    with pytest.raises(ValueError, match="coordinate 'delay' differs"):
        second.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")

    with xr.open_dataset(path, engine=engine) as actual:
        assert actual.sizes["cpi"] == 1


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_requires_complete_record_schema(tmp_path, engine):
    path = tmp_path / "schema.nc"
    first = xr.Dataset({"x": ("cpi", [1.0]), "y": ("cpi", [2.0])}, coords={"cpi": [0]})
    first.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])

    second = xr.Dataset({"x": ("cpi", [3.0])}, coords={"cpi": [1]})
    with pytest.raises(ValueError, match="all existing variables"):
        second.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")

    with xr.open_dataset(path, engine=engine) as actual:
        assert actual.sizes["cpi"] == 1
        np.testing.assert_array_equal(actual["x"], [1.0])
        np.testing.assert_array_equal(actual["y"], [2.0])


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_rejects_new_variable(tmp_path, engine):
    path = tmp_path / "schema.nc"
    first = xr.Dataset({"x": ("cpi", [1.0])}, coords={"cpi": [0]})
    first.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])

    second = xr.Dataset(
        {"x": ("cpi", [2.0]), "new": ("cpi", [3.0])}, coords={"cpi": [1]}
    )
    with pytest.raises(ValueError, match="cannot add new variables"):
        second.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_rejects_metadata_change(tmp_path, engine):
    path = tmp_path / "metadata.nc"
    first = xr.Dataset({"x": ("cpi", [1.0])}, attrs={"kind": "a"})
    first.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])

    second = xr.Dataset({"x": ("cpi", [2.0])}, attrs={"kind": "b"})
    with pytest.raises(ValueError, match="dataset attributes are incompatible"):
        second.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_allows_omitted_existing_metadata(tmp_path, engine):
    path = tmp_path / "metadata-omitted.nc"
    first = xr.Dataset(
        {"x": ("cpi", [1.0], {"units": "V"})},
        coords={"cpi": [0]},
        attrs={"kind": "stream"},
    )
    first.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])

    second = xr.Dataset({"x": ("cpi", [2.0])}, coords={"cpi": [1]})
    second.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")

    with xr.open_dataset(path, engine=engine) as actual:
        np.testing.assert_array_equal(actual["x"], [1.0, 2.0])
        assert actual.attrs["kind"] == "stream"
        assert actual["x"].attrs["units"] == "V"


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_rejects_encoding_kwarg(tmp_path, engine):
    path = tmp_path / "encoding.nc"
    first = xr.Dataset({"x": ("cpi", np.array([1.0], dtype=np.float32))})
    first.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])

    second = xr.Dataset({"x": ("cpi", np.array([2.0], dtype=np.float32))})
    with pytest.raises(ValueError, match="encoding cannot be supplied"):
        second.to_netcdf(
            path,
            engine=engine,
            mode="a",
            append_dim="cpi",
            encoding={"x": {"dtype": "float32"}},
        )


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_requires_existing_file(tmp_path, engine):
    path = tmp_path / "missing.nc"
    ds = xr.Dataset({"x": ("cpi", [1.0])})

    with pytest.raises(FileNotFoundError, match="cannot append to non-existent"):
        ds.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")
    assert not path.exists()


@pytest.mark.parametrize("engine", _available_engines())
def test_to_netcdf_append_dim_cannot_change_unlimited_dimensions(tmp_path, engine):
    path = tmp_path / "unlimited.nc"
    first = xr.Dataset(
        {"x": (("cpi", "delay"), np.ones((1, 2)))},
        coords={"cpi": [0], "delay": [10, 20]},
    )
    first.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])

    second = xr.Dataset(
        {"x": (("cpi", "delay"), np.ones((1, 2)))},
        coords={"cpi": [1], "delay": [10, 20]},
    )
    with pytest.raises(ValueError, match="cannot change which existing"):
        second.to_netcdf(
            path,
            engine=engine,
            mode="a",
            append_dim="cpi",
            unlimited_dims=["cpi", "delay"],
        )


@pytest.mark.parametrize(
    "engine",
    [
        engine
        for engine, module in (("netcdf4", "netCDF4"), ("h5netcdf", "h5netcdf"))
        if importlib.util.find_spec(module) is not None
    ],
)
def test_to_netcdf_append_dim_can_be_nonleading_dimension(tmp_path, engine):
    path = tmp_path / "append-nonleading.nc"
    first = xr.Dataset(
        {"x": (("delay", "cpi"), np.arange(6, dtype=np.float32).reshape(3, 2))},
        coords={"delay": [10, 20, 30], "cpi": [0, 1]},
    )
    first.to_netcdf(path, engine=engine, unlimited_dims=["cpi"])

    second = xr.Dataset(
        {
            "x": (
                ("delay", "cpi"),
                np.arange(6, 12, dtype=np.float32).reshape(3, 2),
            )
        },
        coords={"delay": [10, 20, 30], "cpi": [2, 3]},
    )
    second.to_netcdf(path, engine=engine, mode="a", append_dim="cpi")

    with xr.open_dataset(path, engine=engine) as actual:
        assert actual.sizes == {"delay": 3, "cpi": 4}
        np.testing.assert_array_equal(actual["cpi"], [0, 1, 2, 3])
        np.testing.assert_array_equal(
            actual["x"],
            np.concatenate([first["x"].values, second["x"].values], axis=1),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("netCDF4") is None,
    reason="netCDF4 is not installed",
)
def test_to_netcdf_append_dim_preserves_complex_netcdf4(tmp_path):
    path = tmp_path / "append-complex.nc"
    first = xr.Dataset(
        {"iq": ("cpi", np.array([1 + 2j, 3 + 4j], dtype=np.complex64))},
        coords={"cpi": [0, 1]},
    )
    first.to_netcdf(
        path,
        engine="netcdf4",
        auto_complex=True,
        unlimited_dims=["cpi"],
    )

    second = xr.Dataset(
        {"iq": ("cpi", np.array([5 + 6j, 7 + 8j], dtype=np.complex64))},
        coords={"cpi": [2, 3]},
    )
    second.to_netcdf(
        path,
        engine="netcdf4",
        mode="a",
        auto_complex=True,
        append_dim="cpi",
    )

    with xr.open_dataset(path, engine="netcdf4", auto_complex=True) as actual:
        assert actual["iq"].dtype == np.dtype("complex64")
        np.testing.assert_array_equal(
            actual["iq"],
            np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("dask") is None,
    reason="dask is not installed",
)
def test_to_netcdf_append_dim_rejects_chunked_arrays(tmp_path):
    path = tmp_path / "append-dask.nc"
    first = xr.Dataset({"x": ("cpi", [1.0, 2.0])})
    first.to_netcdf(path, engine="scipy", unlimited_dims=["cpi"])

    second = xr.Dataset({"x": ("cpi", [3.0, 4.0])}).chunk({"cpi": 1})
    with pytest.raises(NotImplementedError, match="chunked arrays"):
        second.to_netcdf(path, engine="scipy", mode="a", append_dim="cpi")
