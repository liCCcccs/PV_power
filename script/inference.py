import logging
from lib.data_loader import DataLoader

logger = logging.getLogger(__name__)


class Inference():
    """ Inference """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s)", self.__class__.__name__, arguments)
        self._args = arguments
        self._data = DataLoader()