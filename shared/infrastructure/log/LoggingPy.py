import logging
import shared.infrastructure.helper.DateTimeHelper as helper

from shared.infrastructure.helper.FileHelper import create_file_log, create_file_log_resume


class LoggingPy:
    def __init__(self, name_model: str = "emptyNameModel", prefix: str = ">>>", log_level: int = 0):
        # Configurar o nível de log
        # (pode ser logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, ou logging.CRITICAL)
        self.logger: logging.Logger
        self.logger_resume: logging.Logger

        # Criar e configurar o logger
        if not name_model:
            name_model = "ERROR"
        self.logger = logging.getLogger(name_model)
        self.logger_resume = logging.getLogger(f'{name_model}_resume')

        log_level = self.__get_log_level(log_level)
        self.logger.setLevel(log_level)
        self.logger_resume.setLevel(logging.INFO)

        # Configurar o manipulador de logs
        self.__config_log_output_file(name_model)
        self.__config_log_output_file_resume(name_model_resume=name_model)
        self.__config_log_output_console(prefix=prefix)

    @staticmethod
    def __get_log_level(log_level: int):
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

        return log_level

    def __config_log_output_file_resume(self, name_model_resume: str):
        # Configurar um manipulador de arquivo
        log_filename = create_file_log_resume(name_model=name_model_resume)
        file_handler = logging.FileHandler(log_filename, mode='w')
        formatter_file = logging.Formatter(f'%(asctime)s - %(message)s')
        formatter_file.converter = lambda *args: helper.get_date_time_now().timetuple()
        file_handler.setFormatter(formatter_file)
        self.logger_resume.addHandler(file_handler)

    def __config_log_output_file(self, name_model: str):
        # Configurar um manipulador de arquivo
        log_filename = create_file_log(name_model=name_model)
        file_handler = logging.FileHandler(log_filename, mode='w')
        formatter_file = logging.Formatter(f'%(asctime)s - %(levelname)s %(message)s')
        formatter_file.converter = lambda *args: helper.get_date_time_now().timetuple()
        file_handler.setFormatter(formatter_file)
        self.logger.addHandler(file_handler)

    def __config_log_output_console(self, prefix: str = ">>>"):
        # Configurar um manipulador de console
        console_handler = logging.StreamHandler()
        formatter_stream = logging.Formatter(f'%(asctime)s - %(levelname)s {prefix} %(message)s',
                                             datefmt='%Y-%m-%d %H:%M')
        formatter_stream.converter = lambda *args: helper.get_date_time_now().timetuple()
        console_handler.setFormatter(formatter_stream)
        self.logger.addHandler(console_handler)

    def log_debug(self, message):
        self.logger.debug(message)

    def log_info(self, message):
        self.logger.info(message)

    def log_resume(self, message):
        self.log_info(message)
        self.logger_resume.info(message)

    def log_error(self, error: Exception):
        self.logger.error(error, exc_info=True)