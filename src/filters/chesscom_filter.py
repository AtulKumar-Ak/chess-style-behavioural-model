#filters/chesscom_filter.py
from src.filters.common import common_filters


MIN_AVG_ELO = 2500


def chesscom_filter(metadata):

    if not common_filters(metadata):
        return False

    avg_elo = (
        metadata["white_elo"]
        +
        metadata["black_elo"]
    ) / 2

    if avg_elo < MIN_AVG_ELO:
        return False

    return True