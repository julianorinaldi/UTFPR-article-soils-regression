import logging
from entityModelConfig import ModelConfig

class loggingPy:
    LOGNAME = "Default"
    def __init__(self, config : ModelConfig):
        # Configurar o nível de log (pode ser logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, ou logging.CRITICAL)
        logging.basicConfig(level=logging.DEBUG)

        # Configurar um manipulador de console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(f'%(asctime)s - %(levelname)s {config.printPrefix} %(message)s')
        console_handler.setFormatter(formatter)

        # Configurar um manipulador de arquivo
        file_handler = logging.FileHandler('logfile.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Criar e configurar o logger
        logger = logging.getLogger(loggingPy.LOGNAME)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    def logDebug(self, message):
        logger = logging.getLogger(loggingPy.LOGNAME)
        logger.debug(message)
        
    def logInfo(self, message):
        logger = logging.getLogger(loggingPy.LOGNAME)
        logger.info(message)