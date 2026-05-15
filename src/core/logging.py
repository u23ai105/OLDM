import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging(level="INFO"):
    logger = logging.getLogger()
    log_handler = logging.StreamHandler(sys.stdout)
    
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    log_handler.setFormatter(formatter)
    
    logger.addHandler(log_handler)
    logger.setLevel(level)
    
    # Disable propagation for quiet libraries
    logging.getLogger("uvicorn.access").propagate = False
    
    return logger

logger = setup_logging()
