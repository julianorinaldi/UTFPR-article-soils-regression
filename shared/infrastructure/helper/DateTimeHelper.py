from datetime import timezone, datetime, timedelta

def get_timezone() -> timezone:
    tz_utc_minus3 = timezone(timedelta(hours=-3))
    return tz_utc_minus3

def get_date_time_now() -> datetime:
    tz_utc_minus3 = get_timezone()
    return datetime.now(tz=tz_utc_minus3)

def get_date_time_str() -> str:
    timestamp = get_date_time_now().strftime("%Y%m%d_%H%M")
    return f'{timestamp}'