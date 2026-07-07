from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_zone_timeseries_clean_area_new.csv"
)

EXPOSURE_PATH = Path(
    "data/metadata/openstudio/apartment_1970/"
    "openstudio_zone_envelope_exposure_summary.csv"
)

OUTPUT_PATH = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_zone_timeseries_clean_area_exposure_new.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)
    exposure = pd.read_csv(EXPOSURE_PATH)

    # Keep only the useful exposure columns.
    exposure = exposure[
        [
            "zone",
            "n_exterior_walls",
            "exterior_wall_gross_area_m2",
            "exterior_wall_net_area_m2",
            "exterior_wall_H_with_film_W_per_K",
            "orientations",
        ]
    ].copy()

    # Normalize names just in case.
    df["zone"] = df["zone"].astype(str).str.strip()
    exposure["zone"] = exposure["zone"].astype(str).str.strip()

    out = df.merge(exposure, on="zone", how="left")

    missing = out["n_exterior_walls"].isna().sum()
    if missing > 0:
        missing_zones = sorted(out.loc[out["n_exterior_walls"].isna(), "zone"].unique())
        raise ValueError(
            f"Missing exposure metadata for {missing} rows. "
            f"Missing zones: {missing_zones}"
        )

    # Area-normalized exposure descriptors.
    out["external_wall_net_area_per_floor_area"] = (
        out["exterior_wall_net_area_m2"] / out["zone_area_m2"]
    )

    out["external_wall_H_per_floor_area"] = (
        out["exterior_wall_H_with_film_W_per_K"] / out["zone_area_m2"]
    )

    # Simple discrete exposure group for robust reference-band matching.
    out["exposure_group"] = (
        out["n_exterior_walls"].astype(int).astype(str) + "_external_wall"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print()
    print(
        out[
            [
                "zone",
                "zone_area_m2",
                "n_exterior_walls",
                "external_wall_net_area_per_floor_area",
                "external_wall_H_per_floor_area",
                "exposure_group",
            ]
        ]
        .drop_duplicates()
        .sort_values(["n_exterior_walls", "zone"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
