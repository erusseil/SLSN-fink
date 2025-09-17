from matplotlib.colors import LinearSegmentedColormap

fink_gradient = LinearSegmentedColormap.from_list('my_gradient', (
    (0.000, (0.082, 0.157, 0.310)),
    (0.500, (0.898, 0.898, 0.898)),
    (1.000, (0.961, 0.384, 0.180))))


full_extraction_path = "/home/etru7215/Documents/minimal_SLSN_pipeline/data/features/full_SLSNe_dataset_features.parquet"
main_path = "/home/etru7215/Documents/minimal_SLSN_pipeline/data/machine_learning/feature_sets/"
extracted_path = main_path + "extracted_SLSNe_dataset_alerts.parquet"
classified_path = main_path + "classified_SLSNe_dataset_alerts.parquet"
unclassified_path = main_path + "unclassified_SLSNe_dataset_alerts.parquet"
hyperparameter_path = "/home/etru7215/Documents/minimal_SLSN_pipeline/data/machine_learning/hyperparameter_search.parquet"
classifier_path = "/home/etru7215/Documents/minimal_SLSN_pipeline/data/machine_learning/SLSN_classifier.joblib"

SLSNI_types = ['SNI-SLSN', 'SNIb-SLSN', 'SNIb-pec-SLSN', 'SNIb/c-SLSN', 'SNIbn-SLSN',
       'SNIc-BL-SLSN', 'SNIc-SLSN']
SLSNII_types = ['SNII-SLSN', 'SNIIP-SLSN', 'SNIIn-SLSN']
SLSN_types = SLSNI_types + SLSNII_types

features_to_use = ['amplitude', 'rise_time', 'fall_time', 'Tmin', 'Tmax', 't_color', 'chi2_rainbow',
                  'snr_amplitude' , 'snr_rise_time' , 'snr_fall_time' , 'snr_Tmin' , 'snr_Tmax' , 'snr_t_color', 'kurtosis', 'max_slope', 
                'z', 'x0', 'x1', 'c', 'chi2_salt', 'flux_amplitude', 'skew', 'distnr', 'duration']
