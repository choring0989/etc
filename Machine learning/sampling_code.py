# Importing data analysis libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def samplePick(df, max):
    np.random.seed(123)
    n = len(df)
    sample = df.take(np.random.permutation(n)[:max])
    return sample

output_file = r'D:\merge_result.csv'

"""
# Take the entire dataset and first sampling -> Uncomment it at the first run
data_names = ["D:\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
              "D:\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
              "D:\Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
              "D:\Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
              "D:\Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
              "D:\Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
              "D:\Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
              "D:\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
              "D:\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
              "D:\Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"]

allData = []

for file in data_names:
    df = pd.read_csv(file)
    # 1st Sampling: 50000 randomly extracted according to the label within one CSV file
    sample_set = df.groupby('Label').apply(samplePick, max=50000)
    allData.append(sample_set)

dataCombine = pd.concat(allData, axis=0, ignore_index=True) # using concat for merge
dataCombine.to_csv(output_file, index=False) # store to_csv
"""

df = pd.read_csv(output_file)
df = df.sample(frac=1).reset_index(drop=True)#Shuffle

# Extract 50000 per attack
sample_data = []

sample_data.append(df.groupby('Label').get_group("Benign")[0:50000])
sample_data.append(df.groupby('Label').get_group("FTP-BruteForce")[0:25000])
sample_data.append(df.groupby('Label').get_group("SSH-Bruteforce")[0:25000])
sample_data.append(df.groupby('Label').get_group("DoS attacks-GoldenEye")[0:13000])
sample_data.append(df.groupby('Label').get_group("DoS attacks-Slowloris")[0:10990])
sample_data.append(df.groupby('Label').get_group("DoS attacks-SlowHTTPTest")[0:13000])
sample_data.append(df.groupby('Label').get_group("DoS attacks-Hulk")[0:13010])
sample_data.append(df.groupby('Label').get_group("Brute Force -Web")[0:611])
sample_data.append(df.groupby('Label').get_group("Brute Force -XSS")[0:230])
sample_data.append(df.groupby('Label').get_group("SQL Injection")[0:87])
sample_data.append(df.groupby('Label').get_group("Infilteration")[0:50000])
sample_data.append(df.groupby('Label').get_group("Bot"))
sample_data.append(df.groupby('Label').get_group("DDoS attacks-LOIC-HTTP")[0:24135])
sample_data.append(df.groupby('Label').get_group("DDOS attack-LOIC-UDP")[0:1730])
sample_data.append(df.groupby('Label').get_group("DDOS attack-HOIC")[0:24135])

sample_data = pd.concat(sample_data, axis=0, ignore_index=True)
print(sample_data.groupby('Label').size())

# Missing value check -> Because there is no missing value, do not proceed
print(sample_data.isnull().sum())

# Removed labeling of infiltration attacks
sample_data = sample_data.replace("Infilteration", "")

print(sample_data.groupby('Label').size())
sample_data.to_csv('D:\\output.csv', index=False) # store to_csv
