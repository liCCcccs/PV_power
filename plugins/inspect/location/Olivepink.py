from ._base import LocationBase


class Location(LocationBase):
    """ Location: Yulara """
    def __init__(self, *args, **kwargs):
        print("Initialized...")
        super().__init__(*args, **kwargs)
        self._feature = {"wind_speed": 0, "temperature": 1, "radiation": 2, "wind_direction": 3, "rainfall": 4,
                         "max_wind_speed": 5, "air_pressure": 6, "hail_accumulation": 7, "pyranometer_1": 8,
                         "temperature_probe_1": 9, "temperature_probe_2": 10, "AEDR": 11, "Active_Power": 12}
        self._inv_feature = {v: k for k, v in self._feature.items()}
        self._date_separator = "[/ :]"
        print("Initialized")
