import logging
from datetime import datetime, timezone, timedelta

class LoggingPy:
    LOGNAME = "Default"
    def __init__(self, nameModel : str = "emptyNameModel", prefix : str = ">>>>>>>>>>>>>>>>>"):
        # Configurar o nível de log (pode ser logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, ou logging.CRITICAL)
        logging.basicConfig(level=logging.INFO)

        tz_utc_minus3 = timezone(timedelta(hours=-3))
        timestamp = datetime.now(tz=tz_utc_minus3).strftime("%Y%m%d_%H%M%S")
        log_filename = f'log_{nameModel}_{timestamp}.log'
        
        # Configurar um manipulador de console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(f'%(asctime)s - %(levelname)s {prefix} %(message)s')

        # Configurar o fuso horário para UTC-3
        tz_utc_minus3 = timezone(timedelta(hours=-3))
        formatter.converter = lambda *args: datetime.now(tz=tz_utc_minus3).timetuple()

        console_handler.setFormatter(formatter)

        # Configurar um manipulador de arquivo
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Criar e configurar o logger
        logger = logging.getLogger(LoggingPy.LOGNAME)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    def logDebug(self, message):
        logger = logging.getLogger(LoggingPy.LOGNAME)
        logger.debug(message)
        
    def logInfo(self, message):
        logger = logging.getLogger(LoggingPy.LOGNAME)
        logger.info(message)