import logging
import datetime
import os


class LogManager:
    def __init__(self, base_dir="output/log", log_filename_prefix="log"):
        self.base_dir = base_dir
        self.log_filename_prefix = log_filename_prefix
        self.setup_directory()
        self.configure_logging()

    def setup_directory(self):
        """确保日志文件目录存在，如果不存在则创建之。"""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def configure_logging(self):
        """配置日志系统，创建文件和控制台日志处理器。"""
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y%m%d_%H%M')
        log_filename = f"{self.base_dir}/{self.log_filename_prefix}_{formatted_time}.txt"

        # 配置文件日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s',
            filename=log_filename,
            filemode='w'  # 每次都创建新文件
        )

        # 配置控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

    def get_logger(self):
        """返回配置好的日志器实例。"""
        return logging.getLogger()