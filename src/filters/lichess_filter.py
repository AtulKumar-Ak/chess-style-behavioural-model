#filters/lichess_filter.py
from src.filters.common import common_filters


MIN_AVG_ELO = 2600


def lichess_filter(metadata):

    if not common_filters(metadata):
        return False

    # ======================================
    # REMOVE BERSERK
    # ======================================

    if metadata["white_berserk"] == "true":
        return False

    if metadata["black_berserk"] == "true":
        return False

    # ======================================
    # REMOVE LOW ELO
    # ======================================

    avg_elo = (
        metadata["white_elo"]
        +
        metadata["black_elo"]
    ) / 2

    if avg_elo < MIN_AVG_ELO:
        return False

    return True