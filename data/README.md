# MetroPT-3 derived data

- Source: UCI Machine Learning Repository, MetroPT-3
- DOI: https://doi.org/10.24432/C5VW3R
- License: CC BY 4.0
- Original authors: Davide Veloso, Luís Ribeiro, João Gama, Catarina Nóbrega,
  Cláudia Pinto, and Marisa Silva
- Original dataset: 1,516,948 compressor telemetry records collected in 2020.

`metropt3_minute.csv.gz` is a derived work: one-minute averages, 60-minute
rolling statistics, chronological split labels, and failure-interval labels
from the dataset documentation. `metropt3_demo.csv` contains two 90-minute
fragments for the UI replay. Generate both files with
`python src/prepare_metropt3.py <official-csv>`.

Redistribution and adaptation follow the source CC BY 4.0 license.
