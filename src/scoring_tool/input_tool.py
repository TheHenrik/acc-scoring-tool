import marimo
app = marimo.App(width="medium")


@app.cell
def imports():
    import plotly.express as px
    import marimo as mo
    import polars as pl
    from pathlib import Path
    return Path, mo, pl, px


@app.cell
def ui_elements(mo):
    # Set up the inputs
    team_input = mo.ui.number(start=1, stop=37, step=1, value=1, label="Team Number:")
    round_input = mo.ui.number(start=1, stop=5, step=1, value=1, label="Round:")
    
    # Wrap them in a mo.ui.dictionary first, then wrap THAT in the form
    settings_form = mo.ui.form(
        element=mo.ui.dictionary({
            "team_id": team_input, 
            "round_num": round_input
        }),
        submit_button_label="Confirm Settings",
        bordered=True
    )
    
    # Display the UI
    ui_card = mo.vstack([
        mo.md("### 🚀 Flight Data Settings"),
        mo.md("Please select the team number and the round below."),
        settings_form
    ])
    
    ui_card
    return round_input, settings_form, team_input, ui_card


@app.cell
def define_switch(mo, settings_form):
    mo.stop(
        not settings_form.value, 
        mo.md("*(Waiting for configuration submission...)*")
    )
    # Create the toggle switch, defaulting to 'True' (Auto-calculate on)
    auto_switch = mo.ui.switch(label="Auto-Calculate", value=True)
    
    return auto_switch,


@app.cell
def setup_data_and_ui(mo, pl, auto_switch, settings_form):
    from scoring_tool.marimo_helper_functions import get_geo_data, get_start_time

    team_id = settings_form.value["team_id"]
    round_num = settings_form.value["round_num"]

    data = pl.read_csv(f"data/rounds/{round_num:02d}_{team_id:02d}_flight-data.csv")
    geo_data = get_geo_data(team_id, round_num)

    # Calculate the suggested/auto start time
    auto_start_time = get_start_time(data, geo_data)

    # Conditionally build the input field based on the switch state
    if auto_switch.value:
        value_input = mo.ui.number(
            value=auto_start_time,
            disabled=True,
            label="Start Time (ms):"
        )
    else:
        value_input = mo.ui.number(
            value=auto_start_time,
            disabled=False,
            label="Start Time (ms):"
        )

    # Place them side-by-side using hstack
    control_layout = mo.hstack(
        [auto_switch, value_input], 
        justify="start", 
        gap=4
    )

    # Display the layout
    control_layout

    return team_id, round_num, data, geo_data, value_input, control_layout


@app.cell
def calculate_penalties_and_time(team_id, round_num, value_input):
    from scoring_tool.marimo_helper_functions import get_current_data
    
    # Extract the live, reactive value from the UI field!
    start_time = value_input.value
    
    # Calculate penalties based on whatever the ACTIVE start time is
    current_penalty, voltage_penalty = get_current_data(team_id, round_num, start_time)
    
    # Passing start_time down to your plot cells
    return start_time, current_penalty, voltage_penalty


@app.cell
def process_current_data(mo, px, team_id, round_num, current_penalty, voltage_penalty, geo_data, data, start_time):
    # --- Velocity Plot ---
    fig_velocity = px.line(
        geo_data, 
        x="Time", 
        y="Velocity_m_s", 
        title="Velocity over Time",
        labels={"Time": "Time (ms)", "Velocity_m_s": "Velocity (m/s)"}
    )
    
    # Add vertical lines to Velocity plot
    fig_velocity.add_vline(x=start_time, line_color="blue", line_dash="dash", annotation_text="Start Time")
    fig_velocity.add_vline(x=start_time + 60_000, line_color="green", line_dash="dash")
    fig_velocity.add_vline(x=start_time + 180_000, line_color="green", line_dash="dash")
    
    # Add horizontal line for 5 km/h (converted to m/s)
    fig_velocity.add_hline(y=5.0 / 3.6, line_color="red", line_dash="solid", annotation_text="5 km/h")

    # --- Current Plot ---
    fig_current = px.line(
        data, 
        x="Time", 
        y="Current", 
        title="Current over Time",
        labels={"Time": "Time (ms)", "Current": "Current (A)"}
    )
    
    # Add vertical lines to Current plot
    fig_current.add_vline(x=start_time, line_color="blue", line_dash="dash", annotation_text="Start Time")
    fig_current.add_vline(x=start_time + 60_000, line_color="green", line_dash="dash")
    fig_current.add_vline(x=start_time + 180_000, line_color="green", line_dash="dash")
    
    # Add horizontal line for 30A limit (Adjust the 'y' value if the limit is different)
    fig_current.add_hline(y=30.0, line_color="red", line_dash="solid", annotation_text="30 A Limit")

    # --- Build UI ---
    current_ui = mo.vstack([
        mo.md("### ⚡ Current Analysis"),
        mo.ui.plotly(fig_current),
        mo.md("### 🏎️ Velocity Analysis"),
        mo.ui.plotly(fig_velocity),
        mo.md(f"**Total Current Penalty:** {current_penalty:.2f} Ampere-seconds"),
        mo.md(f"**Voltage Penalty Applied:** {'Yes' if voltage_penalty else 'No'}")
    ])

    current_ui

    return current_ui, fig_current, fig_velocity


@app.cell
def process_geo_data(mo, pl, team_id, round_num, px, geo_data, start_time):
    from scoring_tool.marimo_helper_functions import get_geo_distance
    
    fig_trajectory = px.line_map(
        geo_data, 
        lat="Latitude", 
        lon="Longitude", 
        hover_data=["Time", "AltitudeGPS", "Velocity_m_s"],
        zoom=14, 
        height=500,
        title=f"Flight Trajectory (Team {team_id:02d}, Round {round_num})"
    )

    fig_height = px.line(
        geo_data, 
        x="Time", 
        y=["AltitudeGPS", "AltitudeBaro"], 
        title="Altitude over Time (GPS vs Barometric)",
        labels={
            "Time": "Time (ms)", 
            "value": "Altitude (m)",   # Plotly defaults to 'value' when plotting multiple columns
            "variable": "Sensor Type"  # Renames the legend title
        }
    )

    distance = get_geo_distance(geo_data, start_time)

    geo_ui = mo.vstack([
        mo.md("### 📊 Flight Analysis"),
        mo.ui.plotly(fig_trajectory),
        mo.ui.plotly(fig_height),
        mo.md(f"**Distance Covered in First 2 Minutes:** {distance:.2f} meters")
    ])
    
    geo_ui
    
    return distance, geo_ui


@app.cell
def setup_score_form(mo, pl, team_id, round_num, distance, current_penalty):
    # Adjust filename if yours doesn't have the 's' at the end
    file_path = f"data/rounds/{round_num:02d}_scores.csv"
    
    try:
        scores_df = pl.read_csv(file_path)
    except Exception:
        # Fallback if the file doesn't exist yet to prevent crashes
        scores_df = pl.DataFrame({
            "ID": [], "Payload": [], "Distance": [], "Current": [], 
            "Loading": [], "Unloading": [], "Penalty": [], "Takeoff": [], 
            "DSQ": [], "Preliminary Score": [], "Score": [], "Logfile": [], "Datetime": []
        })
        
    # Extract the current team's row
    team_row = scores_df.filter(pl.col("ID") == team_id)
    team_data = team_row.to_dicts()[0] if not team_row.is_empty() else {}
    
    # Helper to safely grab existing data or default to an empty string
    def get_val(key, default=""):
        val = team_data.get(key)
        return str(val) if val is not None else default
        
    # Build the input fields, injecting our calculated values for Distance and Current
    score_dict = mo.ui.dictionary({
        "Payload": mo.ui.text(value=get_val("Payload", "0"), label="Payload"),
        "Distance": mo.ui.text(value=str(round(float(distance), 2)), label="Distance (Calculated)"),
        "Current": mo.ui.text(value=str(round(float(current_penalty), 2)), label="Current (Calculated)"),
        "Loading": mo.ui.text(value=get_val("Loading", "0"), label="Loading"),
        "Unloading": mo.ui.text(value=get_val("Unloading", "0"), label="Unloading"),
        "Penalty": mo.ui.text(value=get_val("Penalty", "0"), label="Penalty"),
        "Takeoff": mo.ui.text(value=get_val("Takeoff", ""), label="Takeoff"),
        "DSQ": mo.ui.text(value=get_val("DSQ", ""), label="DSQ"),
        "Preliminary Score": mo.ui.text(value=get_val("Preliminary Score", "0"), label="Preliminary Score"),
        "Score": mo.ui.text(value=get_val("Score", "0"), label="Score"),
        "Logfile": mo.ui.text(value=get_val("Logfile", ""), label="Logfile"),
        "Datetime": mo.ui.text(value=get_val("Datetime", ""), label="Datetime"),
    })
    
    # Wrap it in a form
    score_form = mo.ui.form(score_dict, submit_button_label="💾 Save to CSV", bordered=True)
    
    # Build the display UI
    form_ui = mo.vstack([
        mo.md(f"### 📝 Edit Scores for Team `{team_id:02d}`"),
        mo.md("**Current Data in CSV:**"),
        team_row,
        mo.md("**Update Values:**"),
        score_form
    ])
    
    return file_path, scores_df, score_form, form_ui


@app.cell
def save_scores(mo, pl, file_path, scores_df, score_form, team_id, form_ui):
    # Halt execution until the save button is pressed
    mo.stop(not score_form.value, form_ui)
    
    new_data = score_form.value
    
    # Rebuild a new single-row dictionary containing the ID and the form inputs
    new_row_data = {"ID": [team_id]}
    for k, v in new_data.items():
        new_row_data[k] = [v]
        
    new_row_df = pl.DataFrame(new_row_data)
    
    # Safely cast our text inputs back to the original CSV column datatypes
    for col in new_row_df.columns:
        if col in scores_df.columns:
            target_dtype = scores_df[col].dtype
            
            if target_dtype == pl.Boolean:
                # Custom logic for String -> Boolean conversion
                # Anything matching "true", "1", "t", "yes", or "y" becomes True, everything else is False
                new_row_df = new_row_df.with_columns(
                    pl.col(col).cast(pl.String)
                    .str.to_lowercase()
                    .is_in(["true", "1", "t", "yes", "y"])
                    .fill_null(False)
                )
            else:
                # Normal casting for Floats, Ints, and Strings
                new_row_df = new_row_df.with_columns(
                    pl.col(col).cast(target_dtype, strict=False)
                )
    
    # Ensure the column order matches the original CSV exactly
    new_row_df = new_row_df.select(scores_df.columns)
    
    # Filter out the old row, vertically stack the new row, and re-sort by Team ID
    updated_df = scores_df.filter(pl.col("ID") != team_id).vstack(new_row_df).sort("ID")
    
    # Save back to CSV
    updated_df.write_csv(file_path)
    
    # Output success message and verify the newly saved row
    success_ui = mo.vstack([
        form_ui,
        mo.md("---"),
        mo.md(f"✅ **Success!** Scores for Team `{team_id:02d}` have been updated and saved to `{file_path}`."),
        updated_df.filter(pl.col("ID") == team_id)
    ])
    
    return new_data, new_row_data, new_row_df, updated_df, success_ui


if __name__ == "__main__":
    app.run()
