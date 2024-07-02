import os

from shared.infrastructure.helper.DateTimeHelper import get_date_time_str

def create_file_log_resume(name_model: str) -> str:
    timestamp = get_date_time_str()
    os.makedirs('out/logs_resume', exist_ok=True)
    log_filename = f'out/logs_resume/log_{name_model}_{timestamp}.log'
    return log_filename

def create_file_log(name_model: str) -> str:
    timestamp = get_date_time_str()
    os.makedirs('out/logs', exist_ok=True)
    log_filename = f'out/logs/log_{name_model}_{timestamp}.log'
    return log_filename

def create_file_model(name_model: str, prefix: str = "") -> str:
    timestamp = get_date_time_str()
    os.makedirs('out/model', exist_ok=True)
    filepath_model = f'out/model/{name_model}_{timestamp}.keras'
    if prefix:
        filepath_model = f'out/model/{prefix}_{name_model}_{timestamp}.keras'

    return filepath_model