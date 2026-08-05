from .utils import Logger


def init_from_scratch(configs):
    logger = Logger(configs)


def init_from_pretrained(model_path):
    logger = Logger.from_model_path(model_path)
