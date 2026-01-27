import datetime

# Matrix enumerations
INDEX_COLUMNS = ["origin_id", "destination_id"]
COST_COLUMN = "cost"
ALL_COLUMNS = INDEX_COLUMNS + [COST_COLUMN]
N_DECIMALS = 3

# Defaults that are independent of routing package.
DEFAULT_SPEED_WALKING = 5.0   # km/hr
DEFAULT_SPEED_CYCLING = 18.0   # km/hr
DEFAULT_DEPARTURE_WINDOW = datetime.timedelta(minutes=60)
DEFAULT_TIME_INCREMENT = datetime.timedelta(minutes=1)
ID_COLUMN = "id"

# OTP-only defaults
OTP_DEPARTURE_INCREMENT = datetime.timedelta(minutes=1)

# GTFS modes
GTFS_MODE_MAPPING = {
    0: 'streetcar', 
    1: 'subway', 
    2: 'commuter_rail', 
    3: 'bus', 
    4: 'ferry',
    5: 'cable_tram', 
    6: 'aerial', 
    7: 'funicular', 
    8: 'trolleybus', 
    9: 'monorail',
    # the following are extended GTFS route types
    # https://developers.google.com/transit/gtfs/reference/extended-route-types?visit_id=639046468088128885-3600621307&hl=en&rd=1
    # I've put a good number of them in the following list, which covers the
    # Toronto area. Add more later if needed.
    100: 'commuter_rail',
    106: 'commuter_rail',
    400: 'subway',
    401: 'subway',
    402: 'subway',
    403: 'subway',
    405: 'monorail',
    700: 'bus',
    701: 'bus', 
    702: 'bus',
    704: 'bus',
    900: 'streetcar'
}