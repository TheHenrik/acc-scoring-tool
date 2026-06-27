from cmath import log
from pathlib import Path
import polars as pl


def import_flight_data(file_path: Path, team_id: int, round: int) -> None:
    """Import flight data from the logfile and store the results in the database."""
    logfile = pl.read_csv(
        file_path,
        separator=";",
        has_header=False,
        new_columns=[
            "Datetime",
            "LOC",
            "SatConnected",
            "NumSats",
            "Latitude",
            "_1",
            "Longitude",
            "_2",
            "AltitudeGPS",
            "_3",
            "Pressure",
            "_4",
            "AltitudeBaro",
            "_5",
            "Temperature",
            "_6",
            "Voltage",
            "_7",
            "Current",
            "_8",
            "Power",
            "_9",
        ],
    )
    scores_path = f"data/rounds/{round:02d}_scores.csv"
    scores = pl.read_csv(scores_path)
    datetime = logfile.select("Datetime").head(1).item()
    
    logfile = logfile.with_columns(
        (
            pl.col("Datetime").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%z") - 
            pl.col("Datetime").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%z").first()
        ).dt.total_milliseconds().alias("Time")
    )

    scores = scores.with_columns(
        pl.when(pl.col("ID") == team_id)
        .then(pl.lit(str(file_path)))
        .otherwise(pl.col("Logfile"))
        .alias("Logfile"),
        
        pl.when(pl.col("ID") == team_id)
        .then(pl.lit(datetime))
        .otherwise(pl.col("Datetime"))
        .alias("Datetime")
    )
    
    scores.write_csv(scores_path)
    
    logfile = logfile.select(
        [
            "Time",
            "SatConnected",
            "Latitude",
            "Longitude",
            "AltitudeGPS",
            "AltitudeBaro",
            "Voltage",
            "Current",
        ]
    )
    logfile.write_csv(f"data/rounds/{round:02d}_{team_id:02d}_flight-data.csv")

if __name__ == "__main__":
    import_flight_data(Path("temp/log_10.txt"), team_id=11, round=1)
