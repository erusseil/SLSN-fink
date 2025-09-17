import pandas as pd
import fink_science.ztf.superluminous.slsn_classifier as slsn


data_path = "/home/etru7215/Documents/minimal_SLSN_pipeline/data/light_curves/full_SLSNe_dataset_alerts.parquet"
save_path = "/home/etru7215/Documents/minimal_SLSN_pipeline/data/features/full_SLSNe_dataset_features.parquet"
pdf = pd.read_parquet(data_path)
pdf = slsn.remove_nan(pdf)

# Perform feature extraction
features = slsn.extract_features(pdf)
features['objectId'] = pdf['objectId']
features['label'] = pdf['final_label']
features.to_parquet(save_path)
