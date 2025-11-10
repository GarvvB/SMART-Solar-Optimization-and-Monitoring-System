import numpy as np

def classify_fault(row):
    if row['efficiency']<0.7 and row.get('ghi', 0)>400:
        return "Dust/Soiling"
    elif row.get('ghi',0)<200:
        return 'Low Irradiance'
    elif row.get('clouds_all',0)>70:
        return "Shading/Overcast"
    else:
        return "Normal"
    
def detect_faults(df):
    df['fault_label']=df.apply(classify_fault, axis=1)
    return df