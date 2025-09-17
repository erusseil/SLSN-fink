import pandas as pd
import numpy as np


folder = "/home/etru7215/Documents/minimal_SLSN_pipeline/data/light_curves/"
extra_path = folder + "extra_SLSN.parquet"
main_path = folder + "labeled_RCF_deep_2019_2025.parquet"
save_path = folder + "full_SLSNe_dataset_alerts.parquet"
bad_path = folder + "bad_SLSN.txt"


# We create pseudo alerts. They are splitted light curves but with full history. The assumption is that for ZTF we will be able to reduce the number of alerts each night enough to merge light curves. And for LSST 1 year history should be enough.

def split_to_alerts(ps):
    
    min_points = 7
    min_points_per_band = 3
    n_points = len(ps['cfid'])

    cjd, cmagpsf, csigmapsf, cflux, csigflux, cfid = zip(*sorted(zip(ps["cjd"], ps["cmagpsf"], ps["csigmapsf"],
                                                                     ps["cflux"], ps["csigflux"], ps["cfid"])))
    all_dfs = []
    
    for i in range(max(0, n_points-min_points+1)):

        until = n_points - i 
        sub_cfid = cfid[:until]
        
        # Check that the min point per band is respected
        if all(sum(sub_cfid==band)>=min_points_per_band for band in np.unique(sub_cfid)):
            all_dfs.append(pd.Series(data={"objectId":ps["objectId"], "cjd":cjd[:until],
                                              "cmagpsf":cmagpsf[:until], "csigmapsf":csigmapsf[:until],
                                             "cflux":cflux[:until], "csigflux":csigflux[:until],
                                             "cfid":cfid[:until], "distnr":ps["distnr"], "final_label":ps["final_label"]}))

    return all_dfs


sample1 = pd.read_parquet(main_path)
sample1 = sample1[["objectId", "cjd", "cmagpsf", "csigmapsf", "cflux", "csigflux", "cfid", "distnr", "final_label"]]
sample2 = pd.read_parquet(extra_path)
full_light_curves = pd.concat([sample1, sample2])

bad_SLSNe = list(pd.read_csv(bad_path, header=None)[0])
bad_mask = full_light_curves['objectId'].isin(bad_SLSNe)
full_light_curves = full_light_curves[~bad_mask]

expanded = pd.DataFrame([x for xs in full_light_curves.apply(split_to_alerts, axis=1).to_list() for x in xs])
expanded.to_parquet(save_path)