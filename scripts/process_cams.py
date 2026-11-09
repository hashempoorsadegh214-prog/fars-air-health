#!/usr/bin/env python3
"""Download CAMS PM2.5/PM10 forecasts and create daily Fars products."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cdsapi
import numpy as np
import rasterio
import xarray as xr

from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import mapping, shape


ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_FILE = ROOT / "fars.json"
OUTPUT_DIR = ROOT / "data"
RASTER_DIR = OUTPUT_DIR / "rasters"

DATASET = "cams-global-atmospheric-composition-forecasts"

AREA = ["31.7", "50.3", "27.0", "55.7"]

VARIABLES = {
    "pm25": "particulate_matter_2.5um",
    "pm10": "particulate_matter_10um",
}

ALIASES = {
    "pm25": (
        "pm2p5",
        "pm25",
        "particulate_matter_2.5um",
        "particulate_matter_25um",
    ),
    "pm10": (
        "pm10",
        "particulate_matter_10um",
    ),
}

CYCLES = ("00:00", "12:00")

IRAN_TZ = timezone(timedelta(hours=3, minutes=30))
UTC = timezone.utc


def log(message: str) -> None:
    print(f"[CAMS-FARS] {message}", flush=True)


def requested_date(value: str | None) -> date:
    if value:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("Date must be YYYY-MM-DD.") from exc

    return datetime.now(IRAN_TZ).date() - timedelta(days=1)


def load_boundary() -> list[dict[str, Any]]:

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            "fars.json was not found in the repository."
        )

    with BOUNDARY_FILE.open(encoding="utf-8") as f:
        obj = json.load(f)

    if obj.get("type") == "FeatureCollection":

        raw = [
            x["geometry"]
            for x in obj.get("features", [])
            if x.get("geometry")
        ]

    elif obj.get("type") == "Feature":

        raw = [obj.get("geometry")]

    else:

        raw = [obj]

    result = []

    for item in raw:

        geom = shape(item)

        if not geom.is_valid:
            geom = geom.buffer(0)

        if not geom.is_empty:
            result.append(mapping(geom))

    if not result:
        raise ValueError(
            "No valid polygon geometry was found in fars.json."
        )

    return result


def create_client() -> cdsapi.Client:

    token = os.environ.get("ADS_API_KEY", "").strip()

    if not token:
        raise RuntimeError(
            "GitHub Actions secret ADS_API_KEY is missing."
        )

    if ":" not in token:
        raise RuntimeError(
            "ADS_API_KEY must contain UID:APIKEY."
        )

    return cdsapi.Client(
        url="https://ads.atmosphere.copernicus.eu/api",
        key=token,
        quiet=False,
        progress=True,
    )


def download_cycle(
    client: cdsapi.Client,
    day: date,
    cycle: str,
    folder: Path,
) -> Path:

    tag = cycle.replace(":", "")

    target = folder / f"cams_{day.isoformat()}_{tag}.zip"

    request = {
        "date": [f"{day.isoformat()}/{day.isoformat()}"],
        "time": [cycle],
        "leadtime_hour": [
            str(h) for h in range(0, 121)
        ],
        "type": ["forecast"],
        "data_format": "netcdf_zip",
        "variable": list(VARIABLES.values()),
        "area": AREA,
    }

    log(
        f"Requesting {day.isoformat()} {cycle} UTC forecast."
    )

    result = client.retrieve(DATASET, request)

    result.download(str(target))

    if not target.exists() or target.stat().st_size < 1000:
        raise RuntimeError(
            f"Downloaded archive for {cycle} is empty or too small."
        )

    log(
        f"Downloaded {target.name} "
        f"({target.stat().st_size:,} bytes)."
    )

    return target


def netcdf_files(
    archive: Path,
    folder: Path,
) -> list[Path]:

    out = folder / archive.stem
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out)

    found = sorted(
        p
        for p in out.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (
            ".nc",
            ".nc4",
            ".netcdf",
        )
    )

    if not found:
        raise RuntimeError(
            f"No NetCDF file was found inside {archive.name}."
        )

    return found


def coordinate_name(
    ds: xr.Dataset,
    names: tuple[str, ...],
    standard: str,
) -> str:

    for name in names:

        if name in ds.coords or name in ds.dims:
            return name

    for name, coord in ds.coords.items():

        if (
            str(coord.attrs.get("standard_name", "")).lower()
            == standard
        ):
            return name

    raise RuntimeError(
        f"Could not identify {standard} coordinate. "
        f"Coordinates: {list(ds.coords)}"
    )


def variable_name(
    ds: xr.Dataset,
    pollutant: str,
) -> str:

    for name in ds.data_vars:

        normalized = name.lower().replace("-", "_")

        if normalized in ALIASES[pollutant]:
            return name

    for name, var in ds.data_vars.items():

        desc = " ".join(
            str(var.attrs.get(k, ""))
            for k in (
                "long_name",
                "standard_name",
                "description",
            )
        ).lower()

        if pollutant == "pm25":

            if "pm2.5" in desc or "pm2p5" in desc:
                return name

        if pollutant == "pm10":

            if "pm10" in desc:
                return name

    raise RuntimeError(
        f"Cannot identify {pollutant}; "
        f"variables found: {list(ds.data_vars)}"
    )


def as_datetime(value: Any) -> datetime:

    if isinstance(value, np.datetime64):

        value = (
            value.astype("datetime64[us]")
            .astype(datetime)
        )

    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()

    if not isinstance(value, datetime):

        raise RuntimeError(
            f"Unrecognized valid-time value: {value!r}"
        )

    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)

    return value.astimezone(UTC)


def period_to_hours(value: Any) -> float:

    if isinstance(value, np.timedelta64):

        return float(
            value / np.timedelta64(1, "h")
        )

    if hasattr(value, "total_seconds"):

        return value.total_seconds() / 3600

    return float(value)


def frame_valid_times(
    ds: xr.Dataset,
    init_time: datetime,
    var: xr.DataArray,
) -> tuple[str | None, list[tuple[int, datetime]]]:
    """
    Identify the actual forecast-time dimension.

    CAMS NetCDF may contain:

        forecast_reference_time
        forecast_period
        valid_time

    valid_time may be two-dimensional:
        (forecast_reference_time, forecast_period)

    The singleton reference-time dimension must not be
    mistaken for the hourly forecast dimension.
    """

    # Identify the time dimension from the pollutant variable.
    preferred_dims = (
        "forecast_period",
        "step",
        "leadtime_hour",
        "time",
        "forecast_time",
    )

    time_dim = None

    for candidate in preferred_dims:

        if (
            candidate in var.dims
            and var.sizes[candidate] > 1
        ):
            time_dim = candidate
            break

    # Fallback: identify a non-spatial dimension.
    if time_dim is None:

        spatial_dims = {
            "latitude",
            "longitude",
            "lat",
            "lon",
        }

        candidates = [
            dim
            for dim in var.dims
            if dim not in spatial_dims
            and var.sizes[dim] > 1
        ]

        if len(candidates) == 1:
            time_dim = candidates[0]

        elif len(candidates) > 1:

            raise RuntimeError(
                "Multiple possible forecast dimensions found: "
                f"{candidates}. Variable dimensions: {var.dims}"
            )

    # If the variable has no time dimension, treat it as one frame.
    if time_dim is None:

        if "valid_time" in ds.coords:

            coord = ds["valid_time"].squeeze(drop=True)

            if coord.ndim == 0:

                return None, [
                    (0, as_datetime(coord.values))
                ]

        return None, [(0, init_time)]

    count = var.sizes[time_dim]

    # Prefer explicit valid_time values.
    if "valid_time" in ds.coords:

        coord = ds["valid_time"]

        # Select only the matching forecast period.
        if time_dim in coord.dims:

            pairs = []

            for idx in range(count):

                selected = coord.isel(
                    {time_dim: idx}
                ).squeeze(drop=True)

                if selected.ndim == 0:

                    valid_time = as_datetime(
                        selected.values
                    )

                else:

                    raise RuntimeError(
                        "valid_time still has non-singleton "
                        "dimensions after selecting forecast period: "
                        f"{selected.dims}"
                    )

                pairs.append((idx, valid_time))

            return time_dim, pairs

    # Fallback: calculate valid time from forecast_period.
    if time_dim in ds.coords or time_dim in ds.dims:

        values = np.asarray(
            ds[time_dim].values
        ).reshape(-1)

        if len(values) != count:

            raise RuntimeError(
                "Forecast coordinate length does not match "
                "the pollutant variable."
            )

        pairs = []

        for idx, value in enumerate(values):

            hours = period_to_hours(value)

            valid_time = (
                init_time + timedelta(hours=hours)
            )

            pairs.append((idx, valid_time))

        return time_dim, pairs

    raise RuntimeError(
        "Could not determine forecast valid times. "
        f"Variable dimensions: {var.dims}"
    )


def grid_from_coords(
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:

    lat_order = np.argsort(lat)[::-1]
    lon_order = np.argsort(lon)

    sorted_lat = lat[lat_order]
    sorted_lon = lon[lon_order]

    if len(sorted_lat) < 2 or len(sorted_lon) < 2:

        raise RuntimeError(
            "CAMS grid has fewer than two rows or columns."
        )

    dy = float(
        np.median(np.abs(np.diff(sorted_lat)))
    )

    dx = float(
        np.median(np.abs(np.diff(sorted_lon)))
    )

    transform = from_origin(
        float(sorted_lon.min() - dx / 2),
        float(sorted_lat.max() + dy / 2),
        dx,
        dy,
    )

    grid = {
        "latitude": sorted_lat,
        "longitude": sorted_lon,
        "transform": transform,
        "height": len(sorted_lat),
        "width": len(sorted_lon),
    }

    return grid, lat_order, lon_order


def read_cycle(
    archive: Path,
    day: date,
    cycle: str,
    folder: Path,
    collected: dict[
        tuple[datetime, str],
        np.ndarray
    ],
    existing_grid: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, int]]:

    files = netcdf_files(archive, folder)

    init_time = datetime(
        day.year,
        day.month,
        day.day,
        int(cycle[:2]),
        tzinfo=UTC,
    )

    grid = existing_grid

    counts = {
        "pm25": 0,
        "pm10": 0,
    }

    for nc in files:

        log(f"Inspecting NetCDF: {nc.name}")

        try:

            ds = xr.open_dataset(
                nc,
                decode_times=True,
            )

        except Exception:

            ds = xr.open_dataset(
                nc,
                decode_times=True,
                engine="h5netcdf",
            )

        with ds:

            lat_key = coordinate_name(
                ds,
                ("latitude", "lat"),
                "latitude",
            )

            lon_key = coordinate_name(
                ds,
                ("longitude", "lon"),
                "longitude",
            )

            lat = np.asarray(
                ds[lat_key].values,
                dtype=float,
            ).squeeze()

            lon = np.asarray(
                ds[lon_key].values,
                dtype=float,
            ).squeeze()

            if lat.ndim != 1 or lon.ndim != 1:

                raise RuntimeError(
                    "Expected one-dimensional latitude "
                    "and longitude coordinates."
                )

            if grid is None:

                grid, lat_order, lon_order = (
                    grid_from_coords(lat, lon)
                )

            else:

                lat_order = np.argsort(lat)[::-1]
                lon_order = np.argsort(lon)

                if (
                    len(lat_order) != grid["height"]
                    or len(lon_order) != grid["width"]
                ):

                    raise RuntimeError(
                        "Grid dimensions differ between "
                        "CAMS forecast files."
                    )

                if (
                    not np.allclose(
                        lat[lat_order],
                        grid["latitude"],
                    )
                    or not np.allclose(
                        lon[lon_order],
                        grid["longitude"],
                    )
                ):

                    raise RuntimeError(
                        "Grid coordinates differ between "
                        "CAMS forecast files."
                    )

            for pollutant in VARIABLES:

                vname = variable_name(
                    ds,
                    pollutant,
                )

                var = ds[vname]

                units = str(
                    var.attrs.get("units", "")
                ).lower().replace(" ", "")

                time_dim, times = frame_valid_times(
                    ds,
                    init_time,
                    var,
                )

                log(
                    f"{vname}: dimensions={var.dims}, "
                    f"time_dimension={time_dim}, "
                    f"frames={len(times)}"
                )

                for idx, valid_utc in times:

                    if time_dim and time_dim in var.dims:

                        frame = var.isel(
                            {time_dim: idx}
                        )

                    else:

                        frame = var

                    frame = frame.squeeze(drop=True)

                    for dim in list(frame.dims):

                        if dim not in (
                            lat_key,
                            lon_key,
                        ):

                            if frame.sizes[dim] != 1:

                                raise RuntimeError(
                                    f"Unexpected dimension {dim} "
                                    f"in {vname}. "
                                    f"Remaining dimensions: {frame.dims}"
                                )

                            frame = frame.isel(
                                {dim: 0}
                            )

                    if (
                        lat_key not in frame.dims
                        or lon_key not in frame.dims
                    ):

                        raise RuntimeError(
                            f"{vname} does not contain "
                            "latitude/longitude dimensions. "
                            f"Dimensions: {frame.dims}"
                        )

                    values = np.asarray(
                        frame.transpose(
                            lat_key,
                            lon_key,
                        ).values,
                        dtype=np.float64,
                    )

                    values = values[
                        np.ix_(
                            lat_order,
                            lon_order,
                        )
                    ]

                    # Convert concentration to micrograms/m³.
                    if "kg" in units or not units:

                        values *= 1_000_000_000.0

                    elif "mg" in units:

                        values *= 1000.0

                    elif "µg" in units or "ug" in units:

                        pass

                    else:

                        raise RuntimeError(
                            f"Unrecognized units for "
                            f"{vname}: {units}"
                        )

                    values[~np.isfinite(values)] = np.nan
                    values[values < 0] = np.nan

                    key = (
                        valid_utc,
                        pollutant,
                    )

                    # Later cycle overwrites overlapping valid times.
                    collected[key] = values.astype(
                        np.float32
                    )

                    counts[pollutant] += 1

    if grid is None:

        raise RuntimeError(
            "No grid coordinates were read from the NetCDF."
        )

    return grid, counts


def write_tif(
    path: Path,
    values: np.ndarray,
    grid: dict[str, Any],
    mask: np.ndarray,
) -> None:

    arr = np.where(
        mask & np.isfinite(values),
        values,
        -9999.0,
    ).astype("float32")

    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": grid["transform"],
        "nodata": -9999.0,
        "compress": "deflate",
        "predictor": 3,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(arr, 1)

        dst.set_band_description(
            1,
            "Daily mean surface concentration "
            "(micrograms per cubic metre)",
        )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        help="Forecast initialization date YYYY-MM-DD",
    )

    args = parser.parse_args()

    init_date = requested_date(args.date)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RASTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    boundary = load_boundary()

    client = create_client()

    collected: dict[
        tuple[datetime, str],
        np.ndarray
    ] = {}

    grid = None

    cycle_counts = {}

    with tempfile.TemporaryDirectory(
        prefix="cams_fars_"
    ) as tmp:

        temp = Path(tmp)

        for cycle in CYCLES:

            archive = download_cycle(
                client,
                init_date,
                cycle,
                temp,
            )

            grid, cycle_counts[cycle] = read_cycle(
                archive,
                init_date,
                cycle,
                temp,
                collected,
                grid,
            )

    mask = geometry_mask(
        boundary,
        out_shape=(
            grid["height"],
            grid["width"],
        ),
        transform=grid["transform"],
        invert=True,
        all_touched=True,
    )

    if not mask.any():

        raise RuntimeError(
            "The downloaded CAMS grid does not "
            "intersect the Fars boundary."
        )

    daily_groups: dict[
        date,
        dict[str, list[np.ndarray]]
    ] = {}

    for (valid_time, pollutant), values in collected.items():

        local_day = (
            valid_time.astimezone(IRAN_TZ).date()
        )

        daily_groups.setdefault(
            local_day,
            {},
        ).setdefault(
            pollutant,
            [],
        ).append(values)

    days = []
    quality_days = []

    for local_day in sorted(daily_groups):

        day_data = {
            "date": local_day.isoformat(),
            "pollutants": {},
        }

        q_day = {
            "date": local_day.isoformat(),
            "pollutants": {},
        }

        for pollutant in (
            "pm25",
            "pm10",
        ):

            frames = daily_groups[
                local_day
            ].get(
                pollutant,
                [],
            )

            if not frames:
                continue

            with np.errstate(invalid="ignore"):

                daily_mean = np.nanmean(
                    np.stack(
                        frames,
                        axis=0,
                    ),
                    axis=0,
                )

            valid = (
                mask
                & np.isfinite(daily_mean)
            )

            if not valid.any():
                continue

            vals = daily_mean[valid]

            stats = {
                "mean": round(
                    float(np.mean(vals)),
                    3,
                ),
                "min": round(
                    float(np.min(vals)),
                    3,
                ),
                "max": round(
                    float(np.max(vals)),
                    3,
                ),
                "valid_hour_count": len(frames),
                "unit": "µg/m³",
            }

            filename = (
                f"{pollutant}_"
                f"{local_day.isoformat()}.tif"
            )

            write_tif(
                RASTER_DIR / filename,
                daily_mean,
                grid,
                mask,
            )

            day_data["pollutants"][pollutant] = {
                **stats,
                "geotiff": f"rasters/{filename}",
            }

            q_day["pollutants"][pollutant] = {
                **stats,
                "valid_grid_cells": int(
                    valid.sum()
                ),
            }

        if day_data["pollutants"]:

            days.append(day_data)
            quality_days.append(q_day)

    if not days:

        raise RuntimeError(
            "No daily products were generated. "
            "Check the CAMS response and NetCDF variables."
        )

    payload = {
        "title": "CAMS Air Quality Forecast - Fars Province",
        "source": (
            "Copernicus Atmosphere Monitoring Service "
            "(CAMS) Global Atmospheric Composition Forecasts"
        ),
        "dataset": DATASET,
        "product_type": "forecast",
        "initialization_date": init_date.isoformat(),
        "initialization_cycles_utc": list(CYCLES),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "timezone_for_daily_grouping": "Asia/Tehran (UTC+03:30)",
        "province": "Fars",
        "boundary": "fars.json",
        "native_resolution_degrees_approx": 0.4,
        "concentration_unit": "µg/m³",
        "daily_aggregation": (
            "Mean of available hourly forecast fields grouped "
            "by Iran local calendar date. Later initialization "
            "cycle is preferred for overlapping valid times."
        ),
        "days": days,
        "limitations": [
            "These are model forecasts, not ground-station observations.",
            "The native CAMS grid is approximately 0.4 degrees "
            "(around 40 km); it does not support neighborhood-scale interpretation.",
            "The first and last local dates may contain fewer "
            "than 24 hourly fields.",
            "Resampling does not increase the native spatial resolution.",
        ],
    }

    with (
        OUTPUT_DIR / "air_quality_forecast.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    quality = {
        "status": "success",
        "dataset": DATASET,
        "initialization_date": init_date.isoformat(),
        "cycles_utc": list(CYCLES),
        "request_area_north_west_south_east": AREA,
        "boundary_file": "fars.json",
        "boundary_mask_pixel_count": int(
            mask.sum()
        ),
        "units": "µg/m³",
        "conversion": (
            "kg/m³ × 1,000,000,000 = µg/m³"
        ),
        "cycle_hourly_field_counts": cycle_counts,
        "days": quality_days,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    with (
        OUTPUT_DIR / "quality_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            quality,
            f,
            ensure_ascii=False,
            indent=2,
        )

    log(
        f"Finished: {len(days)} local daily dates processed."
    )

    log("Products saved in data/.")


if __name__ == "__main__":
    main()
