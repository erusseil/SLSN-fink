- labeled_RCF_deep_2019_2025.parquet was obtained by quering all fritz objects that passed the RCF deep filter from 2019 to 2025 using the Fink API. The labels were manually added based on a combination of Fritz labels, TNS labels, and a labelling of SNe brighter than -20 mag.

- On top of that, we use the Fink API to query SLSN from the latest SLSN-I and SLSN-II survey papers. We get them by running the "extra_sample_SLSNe.py" file.

- For now both datasets are full light curves. In order to convert them into a single dataset of alerts, we run the "create_alerts.py" file

