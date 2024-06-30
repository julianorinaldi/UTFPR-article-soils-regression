import logging
import os
from datetime import datetime, timezone, timedelta


class LoggingPy:
    def __init__(self, name_model: str = "emptyNameModel", prefix: str = ">>>", log_level: int = 0):
        # Configurar o nível de log
        # (pode ser logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, ou logging.CRITICAL)
        self.logger: logging.Logger

        if log_level == 0:
            log_level = logging.DEBUG
        elif log_level == 1:
            log_level = logging.INFO
        elif log_level == 2:
            log_level = logging.WARNING
        elif log_level == 3:
            log_level = logging.ERROR
        elif log_level == 4:
            log_level = logging.CRITICAL
        else:
            log_level = logging.DEBUG

        # Criar e configurar o logger
        if not name_model:
            name_model = "ERROR"
        self.logger = logging.getLogger(name_model)
        self.logger.setLevel(log_level)

        tz_utc_minus3 = timezone(timedelta(hours=-3))
        timestamp = datetime.now(tz=tz_utc_minus3).strftime("%Y%m%d_%H%M")
        os.makedirs('out/logs', exist_ok=True)
        log_filename = f'out/logs/log_{name_model}_{timestamp}.log'

        # Configurar um manipulador de arquivo
        file_handler = logging.FileHandler(log_filename, mode='w')
        formatter_file = logging.Formatter(f'%(asctime)s - %(levelname)s %(message)s')
        formatter_file.converter = lambda *args: datetime.now(tz=tz_utc_minus3).timetuple()
        file_handler.setFormatter(formatter_file)
        self.logger.addHandler(file_handler)

        # Configurar um manipulador de console
        console_handler = logging.StreamHandler()
        formatter_stream = logging.Formatter(f'%(asctime)s - %(levelname)s {prefix} %(message)s',
                                             datefmt='%Y-%m-%d %H:%M')
        formatter_stream.converter = lambda *args: datetime.now(tz=tz_utc_minus3).timetuple()
        console_handler.setFormatter(formatter_stream)
        self.logger.addHandler(console_handler)

    def log_debug(self, message):
        self.logger.debug(message)

    def log_info(self, message):
        self.logger.info(message)

    def log_error(self, error: Exception):
        self.logger.error(error, exc_info=True)