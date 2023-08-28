import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',150)

def read_csv_with_progressbar(file_path, **kwargs):
    num_lines = sum(1 for line in open(file_path, encoding='latin-1')) 
    progress_bar = tqdm(total=num_lines)
    chunk_size = 10**6
    data_chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, **kwargs):
        data_chunks.append(chunk)
        progress_bar.update(chunk_size)
    data = pd.concat(data_chunks)
    progress_bar.close()
    return data

%%time

cases = read_csv_with_progressbar('/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/A_TblCase.csv', 
                    sep='\t', 
                    dtype='str', 
                    error_bad_lines=False, 
                    warn_bad_lines=False, 
                    encoding='latin-1', 
                    na_values=' ', 
                    quoting=csv.QUOTE_NONE)

proceedings = read_csv_with_progressbar("/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/B_TblProceeding.csv", 
                          sep='\t', 
                          dtype='str',
                          on_bad_lines='skip',
                          na_values=' ',
                          quoting=csv.QUOTE_NONE)

charges = read_csv_with_progressbar("/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/B_TblProceedCharges.csv", 
                      sep='\t',
                      dtype='str', 
                      na_values=' ',
                      on_bad_lines='skip',
                      quoting=csv.QUOTE_NONE)

charge_lookup = read_csv_with_progressbar("/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/Lookup/tbllookupCharges.csv", 
                            sep='\t',
                            dtype='str',
                            na_values=' ',
                            on_bad_lines='skip',
                            quoting=csv.QUOTE_NONE)
motions = read_csv_with_progressbar("/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/tbl_Court_Motions.csv", 
                            sep='\t',
                            dtype='str',
                            na_values=' ',
                            on_bad_lines='skip',
                            quoting=csv.QUOTE_NONE) 
schedule = read_csv_with_progressbar("/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/tbl_schedule.csv", 
                            sep='\t',
                            dtype='str',
                            na_values=' ',
                            on_bad_lines='skip',
                            quoting=csv.QUOTE_NONE)
case_identifier = read_csv_with_progressbar('/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/A_TblCaseIdentifier.csv', 
                              sep='\t',
                              dtype='str',
                              na_values=' ',
                              on_bad_lines='skip',
                              quoting=csv.QUOTE_NONE)
schema = read_csv_with_progressbar('/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/EOIRDB_Schema.csv', sep='\t',
                              dtype='str',
                              na_values=' ',
                              on_bad_lines='skip',
                              quoting=csv.QUOTE_NONE)

hlcodes = read_csv_with_progressbar('/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/Lookup/tblLookupHloc.csv', sep='\t',
                              dtype='str',
                              na_values=' ',
                              on_bad_lines='skip',
                              quoting=csv.QUOTE_NONE)
case_id_lkup = read_csv_with_progressbar('/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/Lookup/tblLookUpCaseIdentifier.csv', sep='\t',
                              dtype='str',
                              na_values=' ',
                              on_bad_lines='skip',
                              quoting=csv.QUOTE_NONE)
custody_history = read_csv_with_progressbar('/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/tbl_CustodyHistory.csv', sep='\t',
                              dtype='str',
                              na_values=' ',
                              on_bad_lines='skip',
                              quoting=csv.QUOTE_NONE)
base_city_lkup = read_csv_with_progressbar('/Users/stjames/Dropbox/Pablo/data/EOIR TRAC JUNE/Lookup/tblLookupBaseCity.csv', sep='\t',
                              dtype='str',
                              na_values=' ',
                              on_bad_lines='skip',
                              quoting=csv.QUOTE_NONE)


# clean motions table

motions = motions.reset_index()

# Shift column names two spaces to the left

motions.columns = list(motions.columns[2:]) + ['blank 1', 'blank 2']

# clean schedule & proceedings table

# Function to check if a value is numerical
def is_numerical(value):
    if isinstance(value, (int, float)):
        return True
    elif isinstance(value, str) and value.isdigit():
        return True
    return False

# Vectorize the function to work with Pandas Series
vectorized_is_numerical = np.vectorize(is_numerical)

# Create a boolean mask indicating valid IDNCASE values
valid_idncase_mask = vectorized_is_numerical(schedule["IDNCASE"])
valid_idncase_mask_1 = vectorized_is_numerical(proceedings["IDNCASE"])


# Print the invalid IDNCASE values
print("Dropping rows with non-numerical IDNCASE values:")
print(schedule.loc[~valid_idncase_mask, "IDNCASE"])
print(proceedings.loc[~valid_idncase_mask_1, "IDNCASE"])


# Use boolean indexing to keep only rows with numerical IDNCASE values
schedule = schedule[valid_idncase_mask]
proceedings = proceedings[valid_idncase_mask_1]


# convert IDNCASE to int if they are not NA and are digits
proceedings['IDNCASE'] = pd.to_numeric(proceedings['IDNCASE'], errors='coerce').dropna().astype(int)
schedule['IDNCASE'] = pd.to_numeric(schedule['IDNCASE'], errors='coerce').dropna().astype(int)


# convert date values to dt

schedule.INPUT_DATE = pd.to_datetime(schedule.INPUT_DATE, errors='coerce')

# convert IDNCASE to int if they are not NA and are digits
case_identifier['IDNCASE'] = pd.to_numeric(case_identifier['IDNCASE'], errors='coerce').dropna().astype(int)
cases['IDNCASE'] = pd.to_numeric(cases['IDNCASE'], errors='coerce').dropna().astype(int)

print(f"Cases: {cases.shape[0]:,}")
print(f"Proceedings: {proceedings.shape[0]:,}")
print(f"Motions: {motions.shape[0]:,}")
print(f"Schedule: {schedule.shape[0]:,}")


# filter california
# probably need to use BASE_CITY_CODE
CA = base_city_lkup[base_city_lkup['BASE_STATE']=='CA']

# these are my final 4 filters, 2 for proceedings and 2 for cases
# i can ensure the following

# 1) QR appointed after a JCI (excluding QRs w/ missing JCIs
# 2) QR appointed in the same proceeding as JCI
# 3) JCI and QR appointed in the same state (CA)

# selecting proceedings

JPCA = schedule[(schedule['ADJ_RSN']=='62') & (schedule['BASE_CITY_CODE'].isin(CA.BASE_CITY_CODE))]['IDNPROCEEDING'].drop_duplicates().to_frame()
QPCA = schedule[(schedule['ADJ_RSN']=='61') & (schedule['BASE_CITY_CODE'].isin(CA.BASE_CITY_CODE))]['IDNPROCEEDING'].drop_duplicates().to_frame()
QPCA = JPCA[JPCA['IDNPROCEEDING'].isin(QPCA['IDNPROCEEDING'])]

# selecting cases 

JCCA = schedule[(schedule['ADJ_RSN']=='62') & (schedule['BASE_CITY_CODE'].isin(CA.BASE_CITY_CODE))]['IDNCASE'].drop_duplicates().to_frame()
QCCA = schedule[(schedule['ADJ_RSN']=='61') & (schedule['BASE_CITY_CODE'].isin(CA.BASE_CITY_CODE))]['IDNCASE'].drop_duplicates().to_frame()
QCCA = JCCA[JCCA['IDNCASE'].isin(QCCA['IDNCASE'])]



CASES = {'JCCA':JCCA, 'QCCA':QCCA} 

PROCEEDINGS = {'JPCA':JPCA, 'QPCA':QPCA}

for name, df in CASES.items():
    print(f"Case Category: {name}, Number of cases: {df.shape[0]}")

for name, df in PROCEEDINGS.items():
    print(f"Proceeding Category: {name}, Number of proceedings: {df.shape[0]}")