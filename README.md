# cs4641-datacenter

CS 4641: Predicting County-Level Data Center Proliferation
by Jason and Jal

## combine_data.py

Combines data from:

- US Census Bureau : county population 2000-2023
- Census Gazetteer : county land area in sq mi
- EIA Form 861 : state industrial elec rates
- NCSL : state data-center tax exempt
- NOAA 1991-2020 : state avg annual temperature
- USGS 2015 : state freshwater withdrawal
- Census ACS 2019 : county median household income
- Census CBP : NAICS 518210 establishment counts by county/year (target variable)

## pipeline.py

Project part that includes Loading data -> Feature Engineering -> Training -> Evaluation -> Visualization
