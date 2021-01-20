import os
import logging
from plugins.plugin_loader import PluginLoader

# TODO: use plugin load to load inspector
logger = logging.getLogger(__name__)


class Inspect():
    def __init__(self, arguments):
        logging.info("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self._args = arguments
        self._file_path = self._args.file_path
        self._location = self._get_location()
        self._start_date = self._args.start_date
        self._end_date = self._args.end_date
        self._option = self._args.inspect_option

    def process(self):
        logger.debug("Starting inspection...")
        inspector = self._load_inspector()
        self._run_inspection(inspector)

    def _load_inspector(self):
        logger.info("loading inspector...")
        inspector = PluginLoader.get_inspector(self._location)(
            filepath=self._file_path,
            option=self._option,
            start_date=self._start_date,
            end_date=self._end_date
        )
        return inspector

    def _run_inspection(self, inspector):
        inspector.process()
        logger.info("Inspection finished.")

    def _get_location(self):
        if self._file_path[-1] == "/":
            self._file_path = self._file_path[:-1]
        location = os.path.split(self._file_path)[-1]
        return location
