#filters/common.py
MIN_MOVES = 20
MAX_MOVES = 350
def common_filters(metadata):

    # ======================================
    # STANDARD CHESS ONLY
    # ======================================

    if metadata["variant"].lower() != "standard":
        return False

    # ======================================
    # REMOVE BULLET
    # ======================================

    if metadata["speed"] == "bullet":
        return False

    # ======================================
    # REMOVE UNKNOWN
    # ======================================

    if metadata["speed"] == "unknown":
        return False

    # ======================================
    # MINIMUM MOVES
    # ======================================

    if metadata["move_count"] < MIN_MOVES:
        return False
    
    # ======================================
    # MAXIMUM MOVES
    # ======================================

    if metadata["move_count"] > MAX_MOVES:
        return False

    # ======================================
    # UNKNOWN ELO
    # ======================================

    if metadata["white_elo"] == 0:
        return False

    if metadata["black_elo"] == 0:
        return False

    # ======================================
    # ABORTED
    # ======================================

    termination = metadata["termination"].lower()
 
    if termination and "aborted" in termination:
        return False
 
    return True