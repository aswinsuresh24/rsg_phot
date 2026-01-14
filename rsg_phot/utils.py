import logging
import io

class NewlineStdout:
    def __init__(self, stream):
        self.stream = stream

    def write(self, msg):
        self.stream.write(msg.replace('\r', '\n'))
        self.stream.flush()

    def flush(self):
        self.stream.flush()

# logging with tqdm to file
# https://stackoverflow.com/questions/14897756/python-progress-bar-through-logging-module
class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    stdout
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)


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