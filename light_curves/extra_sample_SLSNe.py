import io
import requests
import pandas as pd
import numpy as np
import fink_utils.photometry.conversion as convert


def conv(ps):
    """Convert mag to flux using Fink function"""
    f, ferr = convert.mag2fluxcal_snana(ps['cmagpsf'], ps['csigmapsf'])
    return f, ferr

folder = "/home/etru7215/Documents/minimal_SLSN_pipeline/data/light_curves/"
pessi_path = folder + "SLSNII_pessi_sample.tex"
gomez_path = folder + "gomez_et_al.txt"
save_path = folder + "extra_SLSN.parquet"


########## SLSN-II ############

paper_SLSNII = pd.read_csv("SLSNII_pessi_sample.tex", sep='&', header=None)
all_ids = ["SN "+k[:-1] for k in np.unique(paper_SLSNII[0])]

# Prepare the result table
columns = ["objectId", "cjd", "cmagpsf", "csigmapsf", "cfid", "distnr"]
SLSNII = pd.DataFrame(columns=columns)

for idx in range(len(all_ids)):

    # Search using the TNS name and recover the ZTF name
    r = requests.post(
      'https://api.fink-portal.org/api/v1/resolver',
      json={
        'resolver': 'tns',
        'name': all_ids[idx]})
    names_pdf = pd.read_json(io.BytesIO(r.content)) 

    # Use the ZTF name if it exists
    if len(names_pdf) != 0:
        ZTF_arg = names_pdf['d:internalname'].apply(lambda x: 'ZTF' in x)

        if sum(ZTF_arg) != 0: 
            ZTF_name = names_pdf['d:internalname'][ZTF_arg].iloc[0]
        
            r = requests.post(
              "https://api.fink-portal.org/api/v1/objects",
              json={
                "objectId": ZTF_name,
                "columns": "i:objectId,i:jd,i:magpsf,i:sigmapsf,i:fid,i:jd,d:tag,i:distnr",
                "output-format": "json",
                "withupperlim": "True"
              }
            )
    
            # Format output in a DataFrame
            pdf = pd.read_json(io.BytesIO(r.content))
    
            if len(pdf) != 0:
                pdf = pdf.sort_values('i:jd')
                valid = (pdf['d:tag']=='valid') | (pdf['d:tag']=='badquality')
                pdf = pdf[valid]
                SLSNII.loc[idx] = [ZTF_name, np.array(pdf["i:jd"].values, dtype=float),
                                      np.array(pdf["i:magpsf"].values, dtype=float),
                                      np.array(pdf["i:sigmapsf"].values, dtype=float),
                                      np.array(pdf["i:fid"].values, dtype=int),
                                      np.nanmedian(pdf['i:distnr'])]

########### SLSN-I #################

all_ids = list(pd.read_csv("gomez_et_al.txt", header=None)[0])

# Prepare the result table
SLSNI = pd.DataFrame(columns=columns)

for idx in range(len(all_ids)):
    found = False
    for b4 in ['AT ', '', 'SN ', 'SN', 'AT']:
        if not found:
            # Search using the TNS name and recover the ZTF name
            r = requests.post(
              'https://api.fink-portal.org/api/v1/resolver',
              json={
                'resolver': 'tns',
                'name': b4+all_ids[idx]})
            found = True if r.content != b'[]' else False

    names_pdf = pd.read_json(io.BytesIO(r.content)) 


    # Use the ZTF name if it exists
    if len(names_pdf) != 0:
        ZTF_arg = names_pdf['d:internalname'].apply(lambda x: 'ZTF' in x)

        if sum(ZTF_arg) != 0: 
            ZTF_name = names_pdf['d:internalname'][ZTF_arg].iloc[0]
        
            r = requests.post(
              "https://api.fink-portal.org/api/v1/objects",
              json={
                "objectId": ZTF_name,
                "columns": "i:objectId,i:jd,i:magpsf,i:sigmapsf,i:fid,i:jd,d:tag,i:distnr",
                "output-format": "json",
                "withupperlim": "True"
              }
            )
    
            # Format output in a DataFrame
            pdf = pd.read_json(io.BytesIO(r.content))
    
            if len(pdf) != 0:
                pdf = pdf.sort_values('i:jd')
                valid = (pdf['d:tag']=='valid') | (pdf['d:tag']=='badquality')
                pdf = pdf[valid]
                SLSNI.loc[idx] = [ZTF_name, np.array(pdf["i:jd"].values, dtype=float),
                                      np.array(pdf["i:magpsf"].values, dtype=float),
                                      np.array(pdf["i:sigmapsf"].values, dtype=float),
                                      np.array(pdf["i:fid"].values, dtype=int),
                                      np.nanmedian(pdf['i:distnr'])]

####### MERGE THEM ########

SLSNI['final_label'] = "SNIb/c-SLSN" 
SLSNII['final_label'] = "SNII-SLSN" 

more_SLSN = pd.concat([SLSNI, SLSNII], ignore_index=True)

# We convert the magnitude to flux for the feature extraction.
more_SLSN['cflux'] = None
more_SLSN['csigflux'] = None
more_SLSN[['cflux', 'csigflux']] = more_SLSN.apply(conv, axis=1, result_type='expand')

more_SLSN[['objectId','cjd','cmagpsf','csigmapsf','cflux','csigflux','cfid', 'distnr', 'final_label']].to_parquet(save_path)