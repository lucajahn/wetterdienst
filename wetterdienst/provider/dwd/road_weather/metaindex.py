# -*- coding: utf-8 -*-
# Copyright (c) 2018-2022, earthobservations developers.
# Distributed under the MIT License. See LICENSE for more info.
from datetime import datetime

import pandas as pd
from requests.exceptions import HTTPError

from wetterdienst.exceptions import MetaFileNotFound
from wetterdienst.metadata.columns import Columns
from wetterdienst.util.cache import CacheExpiry
