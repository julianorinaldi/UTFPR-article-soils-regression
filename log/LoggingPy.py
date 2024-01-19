import logging
from datetime import datetime, timezone, timedelta

class LoggingPy:
    def __init__(self, nameModel : str = "emptyNameModel", prefix : str = ">>>>>>>>>>>>>>>>>", log_level : int = 1):
        # Configurar o nível de log (pode ser logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, ou logging.CRITICAL)
        self.logger : logging.Logger
        
        if log_level == 0:
            logLevel=logging.DEBUG
        elif log_level == 1:
            logLevel=logging.INFO
        elif log_level == 2:
            logLevel=logging.WARNING
        elif log_level == 3:
            logLevel=logging.ERROR
        elif log_level == 4:
            logLevel=logging.CRITICAL
        else:
            logLevel=logging.DEBUG
        
        # Criar e configurar o logger
        self.logger = logging.getLogger(nameModel)
        self.logger.setLevel(logLevel)

        tz_utc_minus3 = timezone(timedelta(hours=-3))
        timestamp = datetime.now(tz=tz_utc_minus3).strftime("%Y%m%d_%H%M%S")
        log_filename = f'logs/log_{nameModel}_{timestamp}.log'

        # Configurar um manipulador de arquivo
        file_handler = logging.FileHandler(log_filename)
        formatterFile = logging.Formatter(f'%(asctime)s - %(levelname)s %(message)s')
        formatterFile.converter = lambda *args: datetime.now(tz=tz_utc_minus3).timetuple()
        file_handler.setFormatter(formatterFile)
        self.logger.addHandler(file_handler)

        
        # Configurar um manipulador de console
        console_handler = logging.StreamHandler()
        formatterStream = logging.Formatter(f'%(asctime)s - %(levelname)s {prefix} %(message)s')
        formatterStream.converter = lambda *args: datetime.now(tz=tz_utc_minus3).timetuple()
        console_handler.setFormatter(formatterStream)
        self.logger.addHandler(console_handler)
        

    def logDebug(self, message):
        self.logger.debug(message)
        
    def logInfo(self, message):
        self.logger.info(message)