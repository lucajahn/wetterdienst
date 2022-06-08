# -*- coding: utf-8 -*-
# Copyright (c) 2018-2022, earthobservations developers.
# Distributed under the MIT License. See LICENSE for more info.
import logging
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from wetterdienst import Kind, Period, Provider, Resolution
from wetterdienst.core.scalar.request import ScalarRequestCore
from wetterdienst.core.scalar.values import ScalarValuesCore
from wetterdienst.metadata.columns import Columns
from wetterdienst.metadata.datarange import DataRange
from wetterdienst.metadata.period import PeriodType
from wetterdienst.metadata.resolution import ResolutionType
from wetterdienst.metadata.timezone import Timezone
from wetterdienst.provider.dwd.road_weather.download import (
    download_road_weather_observations_parallel,
)
from wetterdienst.provider.dwd.road_weather.fileindex import (
    create_file_index_for_dwd_road_weather_station,
)
from wetterdienst.provider.dwd.road_weather.metadata.dataset import (
    DwdObservationRoadWeatherDataset,
)
from wetterdienst.provider.dwd.road_weather.metadata.period import (
    DwdRoadWeatherObservationPeriod,
)
from wetterdienst.provider.dwd.road_weather.metadata.station_groups import (
    DwdObservationRoadWeatherStationGroups,
)
from wetterdienst.provider.dwd.road_weather.parser import parse_dwd_road_weather_data
from wetterdienst.util.cache import CacheExpiry
from wetterdienst.util.geo import Coordinates, derive_nearest_neighbours
from wetterdienst.util.network import download_file
from wetterdienst.util.parameter import DatasetTreeCore

log = logging.getLogger(__name__)


class DwdRoadWeatherObservationParameter(DatasetTreeCore):
    """
    enumeration for different parameter/variables
    measured by dwd road weather stations
    """

    class MINUTE_10(Enum):
        ROAD_SURFACE_TEMPERATURE = "roadSurfaceTemperature"
        TEMPERATURE_AIR = "airTemperature"
        ROAD_SURFACE_CONDITION = "roadSurfaceCondition"
        WATER_FILM_THICKNESS = "waterFilmThickness"
        WIND_EXTREME = "maximumWindGustSpeed"
        WIND_EXTREME_DIRECTION = "maximumWindGustDirection"
        WIND_DIRECTION = "windDirection"
        WIND = "windSpeed"
        DEW_POINT = "dewpointTemperature"
        RELATIVE_HUMIDITY = "relativeHumidity"
        TEMPERATURE_SOIL = "soil_temperature"
        VISIBILITY = "visibility"
        PRECIPITATION_TYPE = "precipitationType"
        TOTAL_PRECIPITATION = "totalPrecipitationOrTotalWaterEquivalent"
        INTENSITY_OF_PRECIPITATION = "intensityOfPrecipitation"
        INTENSITY_OF_PHENOMENA = "intensityOfPhenomena"
        HORIZONTAL_VISIBILITY = "horizontalVisibility"
        SHORT_STATION_NAME = "shortStationName"


class DwdRoadWeatherObservationResolution(Enum):
    MINUTE_10 = Resolution.MINUTE_10.value


class DwdRoadWeatherObservationPeriod(Enum):
    HISTORICAL = Period.HISTORICAL.value


class DwdRoadWeatherObservationValues(ScalarValuesCore):
    """
    The DwdRoadWeatherObservationValues class represents a request for
    observation data from road weather stations as provided by the DWD service.
    """

    _data_tz = Timezone.UTC

    def __init__(
        self,
        period: DwdRoadWeatherObservationPeriod = DwdRoadWeatherObservationPeriod.LATEST,
    ):
        self._tz = Timezone.GERMANY
        self.period = period

        self.metaindex = create_meta_index_for_road_weather()

        # actually new stations with group id XX are not available
        self.metaindex = self.metaindex[self.metaindex[Columns.STATION_GROUP.value] != "XX"]

    def __eq__(self, other):
        """Add resolution and periods"""
        return super(DwdRoadWeatherObservationValues, self).__eq__(other)

    def _collect_data_by_rank(
        self,
        latitude: float,
        longitude: float,
        rank: int = 1,
    ) -> pd.DataFrame:
        """
        Takes coordinates and derives the closest stations to the given location.
        These locations define the required Station Groups to download and parse the data
        """
        distances, indices_nearest_neighbours = derive_nearest_neighbours(
            self.metaindex[Columns.LATITUDE.value].values,
            self.metaindex[Columns.LONGITUDE.value].values,
            Coordinates(np.array([latitude]), np.array([longitude])),
            rank,
        )

        self.metaindex = self.metaindex.iloc[indices_nearest_neighbours.ravel(), :]

        required_station_groups = [
            DwdObservationRoadWeatherStationGroups(grp) for grp in self.metaindex[Columns.STATION_GROUP.value].unique()
        ]

        _dat = [
            self._collect_data_by_station_group(road_weather_station_group)
            for road_weather_station_group in required_station_groups
        ]
        road_weather_station_data = pd.concat(
            _dat,
        )

        return road_weather_station_data[
            road_weather_station_data.index.get_level_values("shortStationName").isin(
                self.metaindex[Columns.STATION_ID.value].tolist()
            )
        ]

    def _collect_data_by_station_name(self, station_name: str) -> pd.DataFrame:
        """Takes station_name to download and parse RoadWeather Station data"""
        self.metaindex = self.metaindex.loc[self.metaindex[Columns.NAME.value] == station_name, :]

        required_station_groups = [
            DwdObservationRoadWeatherStationGroups(grp)
            for grp in self.metaindex.loc[:, Columns.STATION_GROUP.value].drop_duplicates()
        ]

        road_weather_station_data = pd.concat(
            [
                self._collect_data_by_station_group(road_weather_station_group)
                for road_weather_station_group in required_station_groups
            ],
        )

        return road_weather_station_data[
            road_weather_station_data.index.get_level_values("shortStationName").isin(
                self.metaindex[Columns.STATION_ID.value].tolist()
            )
        ]

    def _collect_data_by_station_group(
        self,
        road_weather_station_group: DwdObservationRoadWeatherStationGroups,
    ) -> pd.DataFrame:
        """
        Method to collect data for one specified parameter. Manages restoring,
        collection and storing of data, transformation and combination of different
        periods.

        Args:
            road_weather_station_group: subset id for which parameter is collected

        Returns:
            pandas.DataFrame for given parameter of station
        """
        remote_files = create_file_index_for_dwd_road_weather_station(road_weather_station_group)

        if self.period != DwdRoadWeatherObservationPeriod.LATEST:
            raise NotImplementedError("Actually only latest as period type is supported")
        else:
            filenames_and_files = download_road_weather_observations_parallel([remote_files.iloc[-1, 0]])

        return parse_dwd_road_weather_data(filenames_and_files)

    def _coerce_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Use predefined datetime format for given resolution to reduce processing
        time."""
        return

    def _coerce_all_dataframes(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Summarize all dataframes into one general dict"""
        return

    def _create_humanized_parameters_mapping(self) -> Dict[str, str]:
        """Reduce the creation of parameter mapping of the massive amount of parameters
        by specifying the resolution."""
        return

    def _all(self):
        """
        returns stations dataframe

        """
        return self.metaindex

    def _select_required_parameters(
        self,
        data: pd.DataFrame,
        parameters: List[DwdObservationRoadWeatherDataset],
    ) -> pd.DataFrame:
        """This does not work at the moment"""
        return data.loc[:, [param.value for param in parameters]]


class DwdRoadWeatherObservationRequest(ScalarRequestCore):

    provider = Provider.DWD
    kind = Kind.OBSERVATION

    _tz = Timezone.GERMANY

    _values = None

    _has_tidy_data = True

    _has_datasets = True

    _data_range = DataRange.FIXED

    _parameter_base = DwdRoadWeatherObservationParameter
    _unit_tree = None

    _resolution_base = DwdRoadWeatherObservationResolution
    _resolution_type = ResolutionType.FIXED

    _period_base = DwdRoadWeatherObservationPeriod
    _period_type = PeriodType.FIXED

    _base_columns = list(ScalarRequestCore._base_columns)
    _base_columns.extend(
        (
            Columns.STATION_GROUP.value,
            Columns.ROAD_NAME.value,
            Columns.ROAD_SECTOR.value,
            Columns.ROAD_TYPE.value,
            Columns.ROAD_SURFACE_TYPE.value,
            Columns.ROAD_SURROUNDINGS_TYPE.value,
        )
    )

    _endpoint = (
        "https://www.dwd.de/DE/leistungen/opendata/help/stationen/sws_stations_xls.xlsx?__blob=publicationFile&v=11"
    )

    def _all(self) -> pd.DataFrame:
        payload = download_file(self._endpoint, CacheExpiry.METAINDEX)

        df = pd.read_excel(payload)

        print(df)

        df = df.rename(columns={
            "Kennung": Columns.STATION_ID.value,
            "GMA-Name": Columns.NAME.value,
            "Bundesland  ": Columns.STATE.value,
            "Straße / Fahrtrichtung": Columns.ROAD_NAME.value,
            "Strecken-kilometer 100 m": Columns.ROAD_SECTOR.value,
            'Streckenlage (Register "Typen")': Columns.ROAD_SURROUNDINGS_TYPE.value,
            'Streckenbelag (Register "Typen")': Columns.ROAD_SURFACE_TYPE.value,
            "Breite (Dezimalangabe)": Columns.LATITUDE.value,
            "Länge (Dezimalangabe)": Columns.LONGITUDE.value,
            "Höhe in m über NN": Columns.HEIGHT.value,
            "GDS-Verzeichnis": Columns.STATION_GROUP.value,
            "außer Betrieb (gemeldet)": Columns.HAS_FILE.value,
        })

        df = df.loc[df[Columns.HAS_FILE.value].isna() & df[Columns.STATION_GROUP.value] != 0, :]

        df[Columns.LONGITUDE.value] = (
            df[Columns.LONGITUDE.value].replace(",", ".", regex=True).astype(float)
        )
        df[Columns.LATITUDE.value] = (
            df[Columns.LATITUDE.value].replace(",", ".", regex=True).astype(float)
        )

        return df


if __name__ == "__main__":
    request = DwdRoadWeatherObservationRequest("").all()
    print(request.df)
