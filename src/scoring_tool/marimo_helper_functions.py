import math
import polars as pl


def get_current_data(team_id, round_num, start_time):
    c_data = pl.read_csv(f"data/rounds/{round_num:02d}_{team_id:02d}_flight-data.csv")
    
    c_data = c_data.filter(
        pl.col("Time").is_between(start_time, start_time + 180_000)
    )

    # calculate the current penalty: min(1, 0.002 * int(max(0, current - 30))))
    c_data = c_data.with_columns(
        (pl.col("Current") - 30.0).clip(lower_bound=0.0).alias("Excess_Current")
    )

    # Shift columns to get the previous row's values for Trapezoidal math
    c_data = c_data.with_columns([
        pl.col("Time").shift(1).alias("prev_time"),
        pl.col("Excess_Current").shift(1).alias("prev_excess")
    ])

    # Apply the Trapezoidal rule: 0.5 * (height1 + height2) * width
    # We divide by 1000.0 to convert the ms time difference into seconds (Ampere-seconds)
    c_data = c_data.with_columns(
        (
            0.5 * (pl.col("Excess_Current") + pl.col("prev_excess")) * ((pl.col("Time") - pl.col("prev_time")) / 1000.0)
        ).fill_null(0.0).alias("Integral_Step")
    )

    # Sum all the individual areas to get the total integral
    total_excess_integral = c_data.select(pl.sum("Integral_Step")).item()

    # Fallback to 0 if the filtered dataframe happens to be empty
    if total_excess_integral is None: 
        total_excess_integral = 0.0

    total_excess_integral = min(1.0, 0.002 * total_excess_integral)

    current_penalty = total_excess_integral

    # --- 2. Calculate the Voltage Penalty ---
    max_voltage = c_data.select(pl.max("Voltage")).item()
    if max_voltage is None: max_voltage = 0.0
    voltage_penalty = bool(max_voltage > 12.75)

    return current_penalty, voltage_penalty


def get_geo_data(team_id, round_num):    
    data = pl.read_csv(f"data/rounds/{round_num:02d}_{team_id:02d}_flight-data.csv")

    data = smooth_gps_trajectory(data)

    # Earth radius in meters
    R = 6371000.0 

    # Shift columns by 1 to compare current row with previous row
    data = data.with_columns([
        pl.col("Latitude").shift(1).alias("prev_lat"),
        pl.col("Longitude").shift(1).alias("prev_lon"),
        pl.col("Time").shift(1).alias("prev_time")
    ])

    # Haversine formula
    lat_rad = pl.col("Latitude") * math.pi / 180
    prev_lat_rad = pl.col("prev_lat") * math.pi / 180
    d_lat = (pl.col("Latitude") - pl.col("prev_lat")) * math.pi / 180
    d_lon = (pl.col("Longitude") - pl.col("prev_lon")) * math.pi / 180

    a = (d_lat / 2).sin().pow(2) + prev_lat_rad.cos() * lat_rad.cos() * (d_lon / 2).sin().pow(2)

    data = data.with_columns([
        (2 * R * a.sqrt().arcsin()).alias("Distance_m"),
        ((pl.col("Time") - pl.col("prev_time")) / 1000.0).alias("dt_s")
    ])
    
    data = data.with_columns(
        (pl.col("Distance_m") / pl.col("dt_s")).fill_null(0).alias("Velocity_m_s")
    )

    valid_gps_data = data.filter((pl.col("Latitude") != 0) & (pl.col("Longitude") != 0))
    
    return valid_gps_data


def get_start_time(data, geo_data):
    speed_threshold_ms = 5.0 / 3.6
    speed_start = geo_data.filter(pl.col("Velocity_m_s") >= speed_threshold_ms).select(pl.min("Time")).item()


    df_blocks = data.with_columns(
        (pl.col("Current") > 5.0).alias("high_current")
    )
    
    # 2. Assign a unique block_id to each continuous streak of True or False
    df_blocks = df_blocks.with_columns(
        (pl.col("high_current") != pl.col("high_current").shift(1))
        .fill_null(True)
        .cum_sum()
        .alias("block_id")
    )
    
    # 3. Group by the blocks, filter only the high_current blocks, and check duration
    valid_current_blocks = (
        df_blocks.filter(pl.col("high_current"))
        .group_by("block_id")
        .agg([
            pl.min("Time").alias("block_start"),
            pl.max("Time").alias("block_end")
        ])
        .with_columns(
            (pl.col("block_end") - pl.col("block_start")).alias("duration")
        )
        # Keep only blocks that lasted longer than 3000 ms
        .filter(pl.col("duration") > 3000.0) 
    )
    
    # The start time is the exact moment the earliest valid block began
    current_start = valid_current_blocks.select(pl.min("block_start")).item()

    # --- Final Logic: Whichever occurs first ---
    # We put them in a list and filter out 'None' values (in case a condition is never met)
    possible_starts = [t for t in [speed_start, current_start] if t is not None]
    
    if possible_starts:
        start_time = min(possible_starts)
    else:
        # Fallback if the aircraft basically never moved or powered up
        start_time = 0 
    
    return start_time


def smooth_gps_trajectory(df: pl.DataFrame, span: int = 3) -> pl.DataFrame:
    """
    Glättet die Trajektorie rein in Polars mit einem exponentiell gleitenden Durchschnitt.
    
    Args:
        df: Polars DataFrame mit 'Latitude' und 'Longitude'.
        span: Anzahl der Messpunkte, über die gemittelt wird. 
              (Bsp: Bei 10Hz Sensordaten entspricht span=10 einer Glättung über 1 Sekunde).
              Je höher, desto stärker die Glättung.
    """
    valid_df = df.filter((pl.col("Latitude") != 0) & (pl.col("Longitude") != 0))
    
    # Daten müssen chronologisch sortiert sein für EWM
    valid_df = valid_df.sort("Time")
    
    # EWM anwenden (überschreibt die originalen Spalten)
    smoothed_df = valid_df.with_columns([
        pl.col("Latitude").ewm_mean(span=span, ignore_nulls=True).alias("Latitude"),
        pl.col("Longitude").ewm_mean(span=span, ignore_nulls=True).alias("Longitude")
    ])
    
    return smoothed_df


def get_geo_distance(geo_data, start_time):
    dist = geo_data.filter(
        pl.col("Time").is_between(start_time + 60_000, start_time + 180_000)
        ).select(pl.sum("Distance_m")).item()
    return dist
