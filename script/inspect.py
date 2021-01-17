import os
import logging
from plugins.inspect.inspector import Inspector

# TODO: use plugin load to load inspector
logger = logging.getLogger(__name__)


class Inspect():
    def __init__(self, arguments):
        logging.info("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self._args = arguments
        self._file_path = self._args.file_path
        self._start_date = self._args.start_date
        self._end_date = self._args.end_date
        self._option = self._args.inspect_option

    def process(self):
        logger.debug("Starting inspection...")
        if self._start_date is None and self._end_date is None:
            is_full_length = True
        elif self._end_date is None:
            self._end_date = self._start_date
        self._run_inspection()

    def _run_inspection(self):
        inspector = Inspector(self._file_path, self._option, self._start_date, self._end_date)
        inspector.process()

