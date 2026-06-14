import polars as pl

def get_penalties(team_id: int, round_num: int) -> int:
    """ Calculate the penalties for a given team and round. """
    return (round_num * team_id) % 5  # Placeholder implementation


def get_round_score(team_id: int, round_num: int) -> int:
    """ Calculate the score for a given team and round. """
    return round_num + team_id  # Placeholder implementation


def calculate_penalty(team_id: int, round_num: int) -> int:
    """ Calculate the penalty for a given team and round. """
    return (round_num * team_id) % 5  # Placeholder implementation


def calculate_round_score(round_num: int) -> None:
    """ Calculate the score for a given round. """
    pass


def update_scores() -> None:
    """ Update the scores for all teams and rounds. """
    scores = pl.read_csv("data/scores.csv")
    
    for r in range(1, 6):
        calculate_round_score(r)

    round_exprs = [
        pl.col("ID").map_elements(
            lambda team_id, r=r: get_round_score(team_id, r), 
            return_dtype=pl.Int64
        ).alias(f"Round {r}")
        for r in range(1, 6)
    ]

    scores = scores.with_columns(round_exprs)
    
    scores = scores.with_columns(
        pl.concat_list([f"Round {r}" for r in range(1, 6)])
        .list.sort(descending=True)
        .list.head(2)
        .list.sum()
        .alias("Round Total")
    )

    scores.write_csv("data/scores.csv")

def update_penalties() -> None:
    # 1. Read datasets
    scores = pl.read_csv("data/scores.csv")
    penalties = pl.read_csv("data/global_penalties.csv")
    
    # SAFEGUARD 1: If an old "Penalties" column already exists in scores, 
    # drop it so we don't get messy column name collisions during the join.
    if "Penalties" in scores.columns:
        scores = scores.drop("Penalties")
        
    # SAFEGUARD 2: Remove any duplicate IDs from the penalties file.
    # If a team has multiple entries, we sum them up so no data is lost.
    penalties_clean = penalties.group_by("ID").agg(
        pl.col("Total").sum()
    )

    # 3. Perform the Join with Validation
    scores = scores.join(
        penalties_clean.select(["ID", "Total"]), 
        on="ID", 
        how="left",
        validate="m:1" # strict check: guarantees no row duplication in scores
    )
    
    # 4. Safely rename "Total" to "Penalties" and handle perfect records
    scores = scores.with_columns(
        pl.col("Total").fill_null(0).alias("Penalties")
    ).drop("Total")
    
    # 5. Write the validated data back
    scores.write_csv("data/scores.csv")


if __name__ == "__main__":
    update_scores()
    update_penalties()
