import polars as pl
import time


def update_round_score(round_num: int) -> None:
    """Calculate the score for a given round."""
    round_scores = pl.read_csv(f"data/rounds/{round_num:02d}_scores.csv")
    teams = pl.read_csv("data/teams.csv")

    round_scores = round_scores.join(
        teams.select(["ID", "Predicted Payload"]), on="ID", how="left", validate="m:1"
    )

    takeoff_multiplier = pl.when(pl.col("Takeoff") == 40).then(1.15).otherwise(1.0)

    round_scores = round_scores.with_columns(
        (
            takeoff_multiplier
            * pl.col("Payload")
            * (pl.col("Distance") ** 2)
            * (1 - pl.col("Current"))
        ).alias("Preliminary Score")
    )

    payload_bonus = (
        pl.when(pl.col("Payload") == 0)
        .then(0)
        .when(pl.col("Payload") >= pl.col("Predicted Payload"))
        .then(3 * pl.col("Predicted Payload"))
        .when(pl.col("Payload") == (pl.col("Predicted Payload") - 1))
        .then(2 * pl.col("Predicted Payload"))
        .when(pl.col("Payload") == (pl.col("Predicted Payload") - 2))
        .then(1 * pl.col("Predicted Payload"))
        .otherwise(0)
    )

    loading_bonus = (
        pl.when((pl.col("Loading") + pl.col("Unloading")).is_between(1, 120))
        .then(60 * (1 - (pl.col("Loading") + pl.col("Unloading")) / 120))
        .otherwise(0)
    )

    round_scores = round_scores.with_columns(
        (
            pl.when(pl.col("DSQ"))
            .then(0)
            .otherwise(
                (pl.col("Preliminary Score") / pl.col("Preliminary Score").max().clip(1.0) * 1000)
                + loading_bonus
                - pl.col("Penalty")
                + payload_bonus
            )
        ).alias("Score")
    )

    round_scores = round_scores.drop("Predicted Payload")

    round_scores.write_csv(f"data/rounds/{round_num:02d}_scores.csv")


def update_scores(rounds: list[int]) -> None:
    """Update the scores for all teams and rounds."""
    scores = pl.read_csv("data/scores.csv")

    for round_num in rounds:
        update_round_score(round_num)
        round_scores = pl.read_csv(f"data/rounds/{round_num:02d}_scores.csv")

        original_columns = scores.columns

        scores = scores.join(
            round_scores.select(["ID", "Score"]), on="ID", how="left", validate="m:1"
        )

        scores = scores.with_columns(
            pl.col("Score").alias(f"Round {round_num}")
        ).select(original_columns)

    scores = scores.with_columns(
        pl.concat_list([f"Round {r}" for r in range(1, 6)])
        .list.sort(descending=True)
        .list.head(2)
        .list.sum()
        .mul(0.5)
        .cast(pl.Int64)
        .alias("Round Total")
    )

    scores = scores.with_columns(
        (
            pl.col("Round Total")
            + pl.col("Presentation")
            + pl.col("Drawings")
            + pl.col("Report")
            - pl.col("Penalties")
        ).cast(pl.Int64).alias("Total")
    )

    scores.write_csv("data/scores.csv")


def update_penalties() -> None:
    # 1. Read datasets
    scores = pl.read_csv("data/scores.csv")
    penalties = pl.read_csv("data/global_penalties.csv")

    penalties = penalties.with_columns(
        (
            pl.col("Delay_Prelim")
            + pl.col("Delay_Report")
            + pl.col("Poster")
            + pl.col("Drawings")
            + pl.col("Proof_of_flight")
            + pl.col("Tech_inspection")
            + pl.col("Aircraft")
            + pl.col("DSQ")
            + pl.col("Protest")
            + pl.col("Disregard_Instructions")
        ).alias("Total Penalties")
    )

    scores = scores.join(
        penalties.select(["ID", "Total Penalties"]), on="ID", how="left", validate="m:1"
    )

    scores = scores.with_columns(
        pl.col("Total Penalties").fill_null(0).alias("Penalties")
    ).drop("Total Penalties")

    penalties.write_csv("data/global_penalties.csv")
    scores.write_csv("data/scores.csv")


if __name__ == "__main__":
    update_scores(rounds=[1])
    update_penalties()
