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


    if platform == "lichess":

        return headers.get(
            "Variant",
            "Unknown"
        )


    elif platform == "chesscom":


        if "Variant" not in headers:
            return "Standard"

        return headers["Variant"]
    

    elif platform == "pgnmentor":


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


        "white":
            h.get("White", ""),

        "black":
            h.get("Black", ""),


        "white_elo":
            safe_int(h.get("WhiteElo")),

        "black_elo":
            safe_int(h.get("BlackElo")),


        "variant":
            normalize_variant(h, platform),


        "time_control":
            tc,

        "speed":
            classify_time_control(tc),


        "opening":
            opening,

        "eco":
            h.get("ECO", ""),


        "termination":
            h.get("Termination", ""),


        "white_berserk":
            h.get("WhiteBerserk", "false"),

        "black_berserk":
            h.get("BlackBerserk", "false"),


        "move_count":
            len(moves),
    }

    return metadata