import tensorflow as tf

from shared.infrastructure.log.LoggingPy import LoggingPy


class ExecuteProcessBase:
    def __init__(self, logger: LoggingPy):
        self.logger: LoggingPy = logger

    def _get_gpu_strategy(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        # Infos da GPU e Framework
        self.logger.log_debug(f"Quantidade de GPU disponíveis: {physical_devices}")

        # Estratégia para trabalhar com Multi-GPU
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=2))

        return strategy
