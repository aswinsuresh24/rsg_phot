import logging

def getlogger(logfile=None):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(name)s [@ %(asctime)s] [l %(lineno)d] - %(levelname)s - %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename= logfile,
                        filemode='w')

    logger = logging.getLogger("rsgfit")
    logger.setLevel(logging.INFO)
    logger.propagate = False 

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(name)s [@ %(asctime)s] [l %(lineno)d] - %(levelname)s - %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S'
        )
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

logger = getlogger()