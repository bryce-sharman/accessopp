import datetime

# Defaults that are independent of routing package.
DEFAULT_SPEED_WALKING = 5.0   # km/hr
DEFAULT_DEPARTURE_WINDOW = datetime.timedelta(minutes=60)
ID_COLUMN = "id"

# OTP-only defaults
OTP_DEPARTURE_INCREMENT = datetime.timedelta(minutes=1)