# Overview
This report documents the reference pipeline for the mobility workflow. It covers the input snapshots, the processing steps, the artifacts the pipeline writes, and the accessibility and clustering results that the benchmark tasks build on.

# Input data
The pipeline reads fixed local snapshots only: Eurostat-style city indicators and their descriptions, a data.europa.eu catalog sample, a GTFS feed covering the Alicante study area, and Wikidata point-of-interest records. No network access is used at run time, so the inputs are identical on every run.

# Methods
Indicators are selected by theme and pivoted into a city-year matrix. GTFS stops are cleaned for invalid coordinates, duplicate identifiers and missing names. Points of interest are parsed from WKT coordinates and linked to the nearest cleaned stop. The accessibility score combines proximity, departures and route counts, and city-year rows are clustered with KMeans after median imputation and standard scaling.

# Generated outputs
The pipeline writes outputs/t01_selected_indicators.csv, outputs/t05_clean_stops.csv, outputs/t08_pois_clean.csv, outputs/t10_accessibility_scores.csv and outputs/t13_city_clusters.csv, together with the intermediate tables in outputs/ and the two figures in figures/.

# Main results
The accessibility table covers 48 points of interest with scores between 0.142 and 0.865. Clustering assigns 1612 city-year rows across 3 clusters, ranging from 4 to 1598 rows per cluster.

# Limitations
The fixture is a fixed sample of public data and does not represent a production feed. The deterministic checkers verify structural and numerical properties of the artifacts, not the quality of the prose in this report.

# Reproducibility
Regenerate every artifact with `python scripts/prepare_reference_outputs.py`. The script reads only data/inputs, writes to data/reference_outputs, and uses a fixed seed for the clustering step.
