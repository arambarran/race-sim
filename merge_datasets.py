# import libraries
import pandas as pd
from pathlib import Path

OUT_CSV = "Final_Project_Raw.csv"

# normalize race names 
def norm_race_name(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).lower().strip()
    s = s.replace("grand prix", "gp")
    s = " ".join(s.split())
    return s

# load CSV files from "F1 Championship Data (1950-2024)"
def load_core_tables() -> dict:
    tables = {
        "drivers": pd.read_csv("F1-Championship-Data/drivers.csv"),
        "constructors": pd.read_csv("F1-Championship-Data/constructors.csv"),
        "results": pd.read_csv("F1-Championship-Data/results.csv"),
        "races": pd.read_csv("F1-Championship-Data/races.csv"),
        "pit_stops": pd.read_csv("F1-Championship-Data/pit_stops.csv"),
        "circuits": pd.read_csv("F1-Championship-Data/circuits.csv"),
        "status": pd.read_csv("F1-Championship-Data/status.csv"),
    }
    return tables

# load CSV "F1 Pitstop Data (2018-2024)"
def load_enriched() -> pd.DataFrame:
    df = pd.read_csv("F1-Pitstop-Data/f1_pitstops_2018_2024.csv")
    # standardize keys used for joining
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    df["race_key"] = df["Race Name"].map(norm_race_name)
    return df

# build Ferrari-only dataset from results, drivers, races, circuits, status
def build_ferrari_base(t: dict) -> pd.DataFrame:
    ferrari = t["constructors"][t["constructors"]["constructorRef"].str.lower() == "ferrari"]
    ferrari_results = t["results"].merge(ferrari[["constructorId"]], on="constructorId", how="inner")

    ferrari_results = ferrari_results.merge(t["drivers"][["driverId", "forename", "surname", "code"]],on="driverId", how="left")

    ferrari_results = ferrari_results.merge(t["races"][["raceId", "year", "name", "circuitId", "date"]]
                                            .rename(columns={"name": "race_name"}),on="raceId", how="left")

    ferrari_results = ferrari_results.merge(
        t["circuits"][["circuitId", "name", "location", "country"]].rename(columns={"name": "circuit_name"}),on="circuitId", how="left")

    ferrari_results = ferrari_results.merge(t["status"][["statusId", "status"]],on="statusId", how="left")

    # join pit stops (each stop becomes a row)
    strat = ferrari_results.merge(t["pit_stops"][["raceId", "driverId", "stop", "lap", "time", "duration"]],
                                  on=["raceId", "driverId"], how="inner")

    # basic standardization of column names
    strat = strat.rename(columns={
        "forename": "driver_forename",
        "surname": "driver_surname",
        "code": "driver_code",
        "points": "race_points",
        "position": "finish_position",
        "lap": "pit_lap",
        "time": "pit_time",
        "duration": "pit_duration",
    })

    # driver full name and join keys
    strat["Driver"] = (
        strat["driver_forename"].astype(str).str.strip() + " " +
        strat["driver_surname"].astype(str).str.strip()).str.strip()

    strat = strat[(strat["year"] >= 2018) & (strat["year"] <= 2024)]
    strat["race_key"] = strat["race_name"].map(norm_race_name)
    return strat

# define which colums to keep from pitstop dataset
def join_enriched(strat: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "Season","race_key","Driver","Constructor","Laps","Position",
        "TotalPitStops","AvgPitStopTime","Date","Time_of_race","Location","Country",
        "Air_Temp_C","Track_Temp_C","Humidity_%","Wind_Speed_KMH",
        "Lap Time Variation","Total Pit Stops","Tire Usage Aggression","Fast Lap Attempts",
        "Position Changes","Driver Aggression Score",
        "Abbreviation","Stint","Tire Compound","Stint Length","Pit_Lap","Pit_Time"
    ]
    enriched_sub = enriched[[c for c in keep_cols if c in enriched.columns]].copy()

    out = strat.merge(
        enriched_sub,
        left_on=["year", "Driver", "race_key"],
        right_on=["Season", "Driver", "race_key"],
        how="left",
        suffixes=("", "_enriched")
    )
    return out

# organize new dataset columns
def select_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    final_cols = [
        # identifiers & context
        "year","race_name","date","circuit_name","location","country",
        "Driver","driver_code",
        # race outcome
        "finish_position","race_points","status",
        # pit info
        "stop","pit_lap","pit_time","pit_duration",
        # enriched fields (may be NaN when no match)
        "Tire Compound","Stint Length","Stint",
        "Air_Temp_C","Track_Temp_C","Humidity_%","Wind_Speed_KMH",
        "TotalPitStops","AvgPitStopTime",
        "Lap Time Variation","Total Pit Stops","Tire Usage Aggression","Fast Lap Attempts",
        "Position Changes","Driver Aggression Score",
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].sort_values(["year", "race_name", "Driver", "stop"], ascending=True).reset_index(drop=True)
    return df

def main():
    # load raw datasets
    tables = load_core_tables()
    enriched = load_enriched()

    # build Ferrari-only database
    strat = build_ferrari_base(tables)

    # join enriched fields
    joined = join_enriched(strat, enriched)

    # organize dataset
    final_df = select_and_sort(joined)

    # save new CSV
    Path(OUT_CSV).write_text(final_df.to_csv(index=False))
    row_count = len(final_df)
    unique_races = final_df[["year","race_name"]].dropna().drop_duplicates().shape[0]
    unique_drivers = final_df["Driver"].dropna().nunique()

    print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()
