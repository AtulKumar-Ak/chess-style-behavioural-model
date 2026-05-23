#filters/pgnmentor_filter.py
from src.filters.common import (
    common_filters
)

MIN_AVG_ELO = 2300


def pgnmentor_filter(metadata):

    if not common_filters(metadata):
        return False

    avg_elo = (
        metadata["white_elo"]
        +
        metadata["black_elo"]
    ) / 2

    # ======================================
    # OTB ELOS SOMETIMES MISSING
    # ======================================

    if avg_elo != 0:

        if avg_elo < MIN_AVG_ELO:
            return False

    return True