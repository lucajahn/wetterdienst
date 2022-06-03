import json
from enum import Enum
from typing import Optional

from wetterdienst import Provider, Kind, Resolution, Period
from wetterdienst.core.scalar.request import ScalarRequestCore
from wetterdienst.metadata.datarange import DataRange
from wetterdienst.metadata.period import PeriodType
from wetterdienst.metadata.resolution import ResolutionType
from wetterdienst.metadata.timezone import Timezone
from wetterdienst.util.cache import CacheExpiry
from wetterdienst.util.network import FSSPEC_CLIENT_KWARGS, download_file

import pandas as pd

from wetterdienst.util.parameter import DatasetTreeCore

FSSPEC_CLIENT_KWARGS["headers"] = {"User-Agent": "wetterdienst/gutzemann@gmail.com", "Content-Type": "application/json"}


class NwsObservationParameter(DatasetTreeCore):
    class DYNAMIC(Enum):
        PRECIPITATION_HEIGHT = ""


class NwsObservationResolution(Enum):
    DYNAMIC = Resolution.DYNAMIC.value


class NwsObservationRequest(ScalarRequestCore):

    _unit_tree = NwsObservationParameter

    _values = None

    _data_range =  DataRange.FIXED

    _tz = Timezone.USA

    _parameter_base = NwsObservationParameter

    _has_tidy_data = True

    _has_datasets = False

    _period_base = None

    _period_type = PeriodType.FIXED

    _resolution_base = NwsObservationResolution
    _resolution_type = ResolutionType.DYNAMIC

    provider = Provider.NWS
    kind = Kind.OBSERVATION

    _endpoint = "https://api.weather.gov/stations?limit=100"

    def __init__(self, parameter, start_date = None, end_date = None):
        super(NwsObservationRequest, self).__init__(
            parameter=parameter,resolution=Resolution.DYNAMIC, period=Period.HISTORICAL, start_date=start_date, end_date=end_date
        )

    def _all(self) -> pd.DataFrame:
        response = download_file(self._endpoint, CacheExpiry.METAINDEX)

        payload = json.loads(response.read())

        print(payload)

        df = pd.DataFrame.from_dict(payload["features"])

        print(df)


if __name__ == "__main__":
    request = NwsObservationRequest(
        parameter="precipitation_height"
    ).all()


