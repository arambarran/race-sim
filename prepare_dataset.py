# import libraries
import pandas as pd
import numpy as np
import re
import unicodedata
from collections import Counter
from pathlib import Path

# file I/O
IN_CSV  = "Final_Project_Raw.csv"
OUT_CSV = "Final_Project_Cleaned.csv"

def strip_accents(x):
    if pd.isna(x): return x
    return unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii")

def norm_spaces(x):
    if pd.isna(x): return x
    return " ".join(str(x).strip().split())

def to_seconds(val):
    # parse 'M:SS.sss' or 'SS.sss' into seconds
    if pd.isna(val): return np.nan
    s = str(val).strip()
    m = re.match(r"^(?:(\d+):)?(\d+)(?:\.(\d+))?$", s)
    if not m: return np.nan
    mm = float(m.group(1) or 0.0); ss = float(m.group(2))
    ms = float(f"0.{m.group(3)}") if m.group(3) else 0.0
    return mm * 60 + ss + ms

def norm_compound(s):
    if pd.isna(s): return np.nan
    t = str(s).lower().strip()
    if "soft" in t or t in {"s","c5","c4"}: return "Soft"
    if "medium" in t or t in {"m","c3"}:    return "Medium"
    if "hard" in t or t in {"h","c2","c1"}: return "Hard"
    if "inter" in t or "green" in t:        return "Intermediate"
    if "wet" in t or "blue" in t or "full" in t: return "Wet"
    return str(s).title()

def title_keep_gp(s):
    if pd.isna(s): return s
    t = str(s).title()
    t = re.sub(r"\bGp\b", "GP", t)
    t = re.sub(r"\bUsa\b", "USA", t)
    t = re.sub(r"\buk\b",  "UK",  t)
    return t

def race_key_normalize(s):
    # normalize race names for joining
    if pd.isna(s): return s
    t = strip_accents(s).lower().strip()
    t = t.replace("grand prix", "gp")
    return " ".join(t.split())

def norm_key(s):
    if pd.isna(s): return s
    t = strip_accents(s).lower().strip()
    return " ".join(t.split())

def coalesce_dupe_cols(df):
    # collapse duplicate-named columns using first non-null to the left
    for col, cnt in Counter(df.columns).items():
        if cnt > 1:
            same = [c for c in df.columns if c == col]
            merged = df[same].bfill(axis=1).iloc[:, 0]
            df = df.drop(columns=same).assign(**{col: merged})
    return df

def drop_numbered_merge_artifacts(df):
    # drop columns like '4', '5', '4_x', '7_y'
    pattern = re.compile(r"^\d+(_x|_y)?$")
    to_drop = [c for c in df.columns if pattern.match(str(c))]
    return df.drop(columns=to_drop) if to_drop else df

# clean source data
def clean_source(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # basic header fixes
    df.columns = [re.sub(r"[\s/]+", "_", str(c).strip()).lower() for c in df.columns]
    rename_map = {
        "tyre_compound":"tire_compound","stint_len":"stint_length",
        "total_pit_stops":"total_pitstops","avg_pitstoptime":"avg_pit_stop_time",
        "avgpitstoptime":"avg_pit_stop_time","humidity_":"humidity_pct","humidity":"humidity_pct",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    df = coalesce_dupe_cols(df)

    # normalize text fields
    for c in [x for x in ["race_name","circuit_name","location","country","driver","driver_code","tire_compound","stint","status"] if x in df.columns]:
        df[c] = df[c].map(strip_accents).map(norm_spaces)
    if "driver_code" in df.columns:
        df["driver_code"] = df["driver_code"].astype(str).str.upper()
    if "tire_compound" in df.columns:
        df["tire_compound"] = df["tire_compound"].map(norm_compound)

    # to numeric / datetime
    int_like = ["year","finish_position","stop","pit_lap","stint","stint_length","total_pitstops"]
    for c in [x for x in int_like if x in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in [x for x in ["race_points","avg_pit_stop_time","air_temp_c","track_temp_c","humidity_pct","wind_speed_kmh",
                          "lap_time_variation","tire_usage_aggression","fast_lap_attempts","position_changes","driver_aggression_score"] if x in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # minimal derived used downstream
    df["race_key"] = df["race_name"].map(lambda x: title_keep_gp(norm_spaces(strip_accents(x)))) if "race_name" in df.columns else np.nan
    if "year" in df.columns: df["season"] = df["year"]
    if "pit_time" in df.columns: df["pit_time_seconds"] = df["pit_time"].map(to_seconds)
    if "pit_duration" in df.columns: df["pit_duration_seconds"] = df["pit_duration"].map(to_seconds)
    if "stint" in df.columns and "stop" in df.columns: df["stint"] = df["stint"].fillna(df["stop"])
    if "stop" in df.columns: df["is_pit_stop"] = (df["stop"].notna()).astype(int)
    else: df["is_pit_stop"] = 1
    df["ispitstop"] = df["is_pit_stop"].astype(int)
    if "finish_position" in df.columns:
        df["is_podium"] = (df["finish_position"] <= 3).astype(int)
        df["is_points_finish"] = (df.get("race_points", 0).fillna(0) > 0).astype(int)

    # plausibility filters
    df = df.drop_duplicates()
    if "stop" in df.columns: df = df[(df["stop"].isna()) | ((df["stop"] >= 1) & (df["stop"] <= 8))]
    if "pit_lap" in df.columns: df = df[(df["pit_lap"].isna()) | (df["pit_lap"] >= 1)]
    if "pit_duration_seconds" in df.columns:
        df.loc[df["pit_duration_seconds"] <= 0, "pit_duration_seconds"] = np.nan
        df = df[(df["pit_duration_seconds"].isna()) | ((df["pit_duration_seconds"] >= 1) & (df["pit_duration_seconds"] <= 60))]
    if "finish_position" in df.columns:
        bad_pos = (df["finish_position"] < 1) | (df["finish_position"] > 20)
        df.loc[bad_pos, "finish_position"] = np.nan
    if "race_points" in df.columns:
        df.loc[(df["race_points"] < 0) | (df["race_points"] > 30), "race_points"] = np.nan

    # impute numeric by season median then global
    for col in [x for x in ["race_points","finish_position","pit_duration_seconds","pit_time_seconds",
                            "air_temp_c","track_temp_c","humidity_pct","wind_speed_kmh",
                            "avg_pit_stop_time","stint_length","total_pitstops"] if x in df.columns]:
        if pd.api.types.is_numeric_dtype(df[col]):
            if "season" in df.columns:
                df[col] = df.groupby("season")[col].transform(lambda s: s.fillna(s.median()))
            df[col] = df[col].fillna(df[col].median())

    # impute categoricals by mode
    for col in [x for x in ["tire_compound","driver","driver_code","location","country","status"] if x in df.columns]:
        mode = df[col].dropna().mode()
        if not mode.empty: df[col] = df[col].fillna(mode.iloc[0])

    # drop impossible stint length rows (== 0) and final strict dropna
    if "stint_length" in df.columns: df = df[~(df["stint_length"] == 0)]
    df = df.dropna(how="any")

    # cast ints safely
    for c in [x for x in int_like if x in df.columns]: df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype(int)
    if "season" in df.columns: df["season"] = pd.to_numeric(df["season"], errors="coerce").round(0).astype(int)
    for c in ["is_pit_stop","ispitstop","is_podium","is_points_finish"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype(int)

    # reorder then title-case headers to PascalCase-like
    preferred = [x for x in [
        "season","year","date","race_name","race_key","circuit_name","location","country",
        "driver","driver_code","finish_position","race_points","is_podium","is_points_finish","status",
        "stop","pit_lap","tire_compound","stint","stint_length","ispitstop","is_pit_stop",
        "pit_duration","pit_duration_seconds","pit_time","pit_time_seconds",
        "air_temp_c","track_temp_c","humidity_pct","wind_speed_kmh",
        "total_pitstops","avg_pit_stop_time","lap_time_variation",
        "tire_usage_aggression","fast_lap_attempts","position_changes","driver_aggression_score"
    ] if x in df.columns]
    df = df[[*preferred, *[c for c in df.columns if c not in preferred]]]
    name_map = {
        "year":"Year","season":"Season","date":"Date","race_name":"RaceName","race_key":"RaceKey",
        "circuit_name":"CircuitName","location":"Location","country":"Country","driver":"Driver",
        "driver_code":"DriverCode","finish_position":"FinishPosition","race_points":"RacePoints","status":"Status",
        "is_podium":"IsPodium","is_points_finish":"IsPointsFinish","stop":"Stop","pit_lap":"PitLap",
        "tire_compound":"TireCompound","stint":"Stint","stint_length":"StintLength","is_pit_stop":"IsPitStop",
        "ispitstop":"IsPitStop","pit_duration":"PitDuration","pit_duration_seconds":"PitDurationSeconds",
        "pit_time":"PitTime","pit_time_seconds":"PitTimeSeconds","air_temp_c":"AirTemp","track_temp_c":"TrackTemp",
        "humidity_pct":"HumidityPct","wind_speed_kmh":"WindSpeedKMH","total_pitstops":"TotalPitStops",
        "avg_pit_stop_time":"AvgPitStopTime","lap_time_variation":"LapTimeVariation",
        "tire_usage_aggression":"TireUsageAggression","fast_lap_attempts":"FastLapAttempts",
        "position_changes":"PositionChanges","driver_aggression_score":"DriverAggressionScore",
    }
    df.columns = [name_map.get(c, "".join(w.capitalize() for w in c.split("_"))) for c in df.columns]
    return df

# derive features
def transform_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()

    # coalesce duplicate columns, keep any IsPitStop as max
    for col, cnt in Counter(df.columns).items():
        if cnt > 1:
            same = [c for c in df.columns if c == col]
            merged = df[same].bfill(axis=1).iloc[:, 0]
            if col == "IsPitStop":
                merged = df[same].max(axis=1)
            df = df.drop(columns=same).assign(**{col: merged})

    # derive strategy metrics keyed by Season/RaceName/Driver
    if all(k in df.columns for k in ["Season","RaceName","Driver"]):
        key = ["Season","RaceName","Driver"]
        g = df.sort_values(key + ["Stint"]).groupby(key)

        if "TireCompound" in df.columns:
            df["OpeningCompound"] = g["TireCompound"].transform("first")
            df["LastCompound"]    = g["TireCompound"].transform("last")
        if "Stop" in df.columns:
            df["NumStops"] = g["Stop"].transform("max")
        if "StintLength" in df.columns:
            df["AvgStintLength"] = g["StintLength"].transform("mean")

        # pit-loss metrics
        if "PitDurationSeconds" in df.columns:
            race_med = df.groupby(["Season","RaceName"])["PitDurationSeconds"].transform("median")
            df["StopRelativePitLossSeconds"] = df["PitDurationSeconds"] - race_med
            df["DriverAvgPit"] = g["PitDurationSeconds"].transform("mean")
            race_avg = df.groupby(["Season","RaceName"])["PitDurationSeconds"].transform("mean")
            df["DriverAvgRelativePitLossSeconds"] = df["DriverAvgPit"] - race_avg
            df = df.drop(columns=["DriverAvgPit"])

        # first/second pit lap
        if "PitLap" in df.columns:
            df["FirstPitLap"] = g["PitLap"].transform("min")
            df["_rank"] = g["PitLap"].rank(method="first")
            second = df[df["_rank"] == 2][key + ["PitLap"]].rename(columns={"PitLap":"SecondPitLap"})
            df = df.merge(second, on=key, how="left")
            df = df.drop(columns=["_rank"])
        else:
            df["FirstPitLap"] = np.nan
            df["SecondPitLap"] = np.nan

        # strategy type label
        if "NumStops" in df.columns:
            df["StrategyType"] = df["NumStops"].map({0:"NoStop",1:"1-stop",2:"2-stop",3:"3-stop"}).fillna("3+-stop").astype(str)

        # stint compounds (restrict to 1–3)
        if all(c in df.columns for c in ["Stint","TireCompound"]):
            s_comp = (df.sort_values(key+["Stint"])
                        .drop_duplicates(subset=key+["Stint"])
                        [key+["Stint","TireCompound"]])
            s_comp = s_comp[s_comp["Stint"].isin([1,2,3])]
            wide_c = (s_comp.pivot(index=key, columns="Stint", values="TireCompound")
                            .rename(columns={1:"Stint1Compound",2:"Stint2Compound",3:"Stint3Compound"})
                            .reset_index())
            df = df.merge(wide_c, on=key, how="left")
        for c in ["Stint1Compound","Stint2Compound","Stint3Compound"]:
            if c not in df.columns: df[c] = "Unknown"
            df[c] = df[c].astype(str).replace({"nan": np.nan}).fillna("Unknown")

        # stint lengths (restrict to 1–3)
        if all(c in df.columns for c in ["Stint","StintLength"]):
            s_len = (df.sort_values(key+["Stint"])
                       .drop_duplicates(subset=key+["Stint"])
                       [key+["Stint","StintLength"]])
            s_len = s_len[s_len["Stint"].isin([1,2,3])]
            wide_l = (s_len.pivot(index=key, columns="Stint", values="StintLength")
                           .rename(columns={1:"Stint1Length",2:"Stint2Length",3:"Stint3Length"})
                           .reset_index())
            df = df.merge(wide_l, on=key, how="left")
        for c in ["Stint1Length","Stint2Length","Stint3Length"]:
            if c not in df.columns: df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round(0).astype(int)

    # context flags
    if "TrackTemp" in df.columns: df["HotRaceFlag"] = (df["TrackTemp"] > 35).astype("Int8")
    if "HumidityPct" in df.columns: df["RainRiskFlag"] = (df["HumidityPct"] >= 70).astype("Int8")

    # standardize decimals
    if "AvgStintLength" in df.columns: df["AvgStintLength"] = pd.to_numeric(df["AvgStintLength"], errors="coerce").round(1)
    for c in [x for x in ["AirTemp","TrackTemp","WindSpeedKMH"] if x in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(1)
    if "HumidityPct" in df.columns:
        df["HumidityPct"] = pd.to_numeric(df["HumidityPct"], errors="coerce").round(0).fillna(0).astype(int)
    for c in [x for x in [
        "PitDurationSeconds","PitTimeSeconds","StopRelativePitLossSeconds","DriverAvgRelativePitLossSeconds",
        "AvgPitStopTime","LapTimeVariation"
    ] if x in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
    for c in [x for x in [
        "Season","Year","FinishPosition","Stop","PitLap","Stint","StintLength","TotalPitStops",
        "NumStops","IsPodium","IsPointsFinish","IsPitStop","HighDegTrackFlag","HotRaceFlag","RainRiskFlag",
        "FirstPitLap","SecondPitLap"
    ] if x in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(0).fillna(0).astype(int)

    # final safeguard against numbered merge artifacts
    df = drop_numbered_merge_artifacts(df)
    return df

# add grid position (ergast) + track type
def load_ergast() -> dict:
    # load local ergast tables
    return {
        "drivers": pd.read_csv("F1-Championship-Data/drivers.csv"),
        "results": pd.read_csv("F1-Championship-Dataresults.csv"),
        "races":   pd.read_csv("F1-Championship-Dataraces.csv"),
    }

def classify_from_circuit(cn: str) -> str:
    # map circuit name to coarse track type
    if not isinstance(cn, str): return "Unknown"
    key = norm_key(cn)
    if any(k in key for k in [
        "baku city circuit","jeddah","marina bay","miami international autodrome",
        "las vegas strip","circuit gilles villeneuve","albert park","monaco"
    ]): return "Street/Semi-Street"
    if any(k in key for k in [
        "circuit de barcelona","catalunya","bahrain international","silverstone","suzuka",
        "zandvoort","losail","lusail","circuit of the americas","cota","paul ricard"
    ]): return "High-Degradation"
    if any(k in key for k in [
        "autodromo nazionale di monza","spa-francorchamps","circuit de spa","hermanos rodriguez",
        "red bull ring","monza","spa","mexico"
    ]): return "Power-Sensitive"
    if any(k in key for k in [
        "hungaroring","autodromo enzo e dino ferrari","imola","sochi autro","sochi autodom","sochi autrodrom",
        "valencia","shanghai international","yas marina","interlagos","jose carlos pace"
    ]): return "Technical/Traction-Limited"
    return "Other"

# run pipeline
def main():
    # load input dataset
    raw = pd.read_csv(IN_CSV)
    if "ispitstop" in raw.columns:
        raw = raw.drop(columns=["ispitstop"])

    # clean + transform (v9 rules)
    clean = clean_source(raw)
    model = transform_features(clean)

    # drop extra features not needed (per request)
    model = model.drop(columns=[c for c in ["FastLapAttempts","PositionChanges"] if c in model.columns], errors="ignore")

    # hold current column order; we'll append at the end
    base_cols = model.columns.tolist()

    # build join keys for ergast merge
    work = model.copy()
    work["Season"] = work["Season"] if "Season" in work.columns else work.get("Year", np.nan)
    work["__race_key_norm"] = work["RaceName"].apply(race_key_normalize)
    work["__driver_norm"] = work["Driver"].apply(lambda s: strip_accents(str(s)).strip())

    # join grid positions from ergast
    try:
        e = load_ergast()
        drivers = e["drivers"].copy()
        results = e["results"].copy()
        races   = e["races"].copy()

        drivers["__driver_full"] = (drivers["forename"].astype(str).str.strip() + " " + drivers["surname"].astype(str).str.strip())
        drivers["__driver_norm"] = drivers["__driver_full"].apply(lambda s: strip_accents(str(s)).strip())

        res = results.merge(drivers[["driverId","__driver_norm"]], on="driverId", how="left")
        res = res.merge(races[["raceId","year","name"]], on="raceId", how="left")
        res["Season"] = res["year"]
        res["__race_key_norm"] = res["name"].apply(race_key_normalize)
        res["GridPosition"] = pd.to_numeric(res["grid"], errors="coerce")

        grid_map = (res[["Season","__race_key_norm","__driver_norm","GridPosition"]]
                    .dropna(subset=["GridPosition"])
                    .drop_duplicates())

        work = work.merge(grid_map, on=["Season","__race_key_norm","__driver_norm"], how="left")
    except Exception:
        work["GridPosition"] = np.nan

    # classify track type from circuit name
    work["TrackType"] = work["CircuitName"].apply(classify_from_circuit)

    # append new columns at the end, keep everything else identical
    for col in ["GridPosition","TrackType"]:
        if col not in model.columns:
            model[col] = work[col].values
    model = model[base_cols + [c for c in ["GridPosition","TrackType"] if c in model.columns and c not in base_cols]]

    # save csv
    Path(OUT_CSV).write_text(model.to_csv(index=False))
    print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()
