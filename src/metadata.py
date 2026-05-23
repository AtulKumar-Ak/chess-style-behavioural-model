#src/metadata.py
def safe_int(x):

    try:
        return int(x)

    except:
        return 0


def classify_time_control(tc):

    try:

        if tc == "-":
            return "unknown"

        if "+" in tc:

            base, increment = tc.split("+")
            seconds = int(base)

        else:

            seconds = int(tc)

        minutes = seconds / 60

        if minutes < 2:
            return "bullet"

        elif minutes < 10:
            return "blitz"

        elif minutes < 30:
            return "rapid"

        else:
            return "classical"

    except:
        return "unknown"


def normalize_variant(headers, platform):

    # ==========================================
    # LICHESS
    # ==========================================

    if platform == "lichess":

        return headers.get(
            "Variant",
            "Unknown"
        )

    # ==========================================
    # CHESS.COM
    # ==========================================

    elif platform == "chesscom":

        # --------------------------------------
        # Missing Variant => Standard
        # --------------------------------------

        if "Variant" not in headers:
            return "Standard"

        return headers["Variant"]
    
    # ==========================================
    # PGN MENTOR
    # ==========================================

    elif platform == "pgnmentor":

        # --------------------------------------
        # Missing Variant => Standard
        # --------------------------------------

        if "Variant" not in headers:
            return "Standard"

        return headers["Variant"]
    

    return "Unknown"


def extract_metadata(game, platform):

    h = game.headers

    if platform == "pgnmentor":

        tc = h.get("TimeControl", "5400")
        if not tc or tc == "?":
            tc = "5400"

    else:

        tc = h.get(
            "TimeControl",
            ""
        )

    moves = list(game.mainline_moves())
    # ======================================
    # OPENING
    # ======================================

    opening = h.get("Opening", "")

    # Chess.com fallback via ECOUrl
    if not opening:

        eco_url = h.get("ECOUrl", "")

        if eco_url:

            opening = (
                eco_url
                .split("/")[-1]
                .replace("-", " ")
            )

    opening = opening.strip()

    metadata = {

        # ======================================
        # PLAYERS
        # ======================================

        "white":
            h.get("White", ""),

        "black":
            h.get("Black", ""),

        # ======================================
        # ELOS
        # ======================================

        "white_elo":
            safe_int(h.get("WhiteElo")),

        "black_elo":
            safe_int(h.get("BlackElo")),

        # ======================================
        # VARIANT
        # ======================================

        "variant":
            normalize_variant(h, platform),

        # ======================================
        # TIME CONTROL
        # ======================================

        "time_control":
            tc,

        "speed":
            classify_time_control(tc),

        # ======================================
        # OPENING
        # ======================================

        "opening":
            opening,

        "eco":
            h.get("ECO", ""),

        # ======================================
        # TERMINATION
        # ======================================

        "termination":
            h.get("Termination", ""),

        # ======================================
        # BERSERK
        # ======================================

        "white_berserk":
            h.get("WhiteBerserk", "false"),

        "black_berserk":
            h.get("BlackBerserk", "false"),

        # ======================================
        # MOVE COUNT
        # ======================================

        "move_count":
            len(moves),
    }

    return metadata