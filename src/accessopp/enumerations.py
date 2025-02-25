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
    0: 'streetcar', 1: 'subway', 2: 'commuter_rail', 3: 'bus', 4: 'ferry',
    5: 'cable_tram', 6: 'aerial', 7: 'funicular', 8: 'trolleybus', 
    9: 'monorail'
}