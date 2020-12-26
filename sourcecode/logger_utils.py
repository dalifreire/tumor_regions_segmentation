import logging

formatter = logging.Formatter('%(asctime)s :: %(levelname)s %(funcName)s :: %(message)s')

# file handler
fh = logging.FileHandler('application.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

# console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(ch)
logger.addHandler(fh)
