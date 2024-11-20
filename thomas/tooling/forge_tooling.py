def disable_forge_logger():
    from loguru import logger
    logger.disable("")