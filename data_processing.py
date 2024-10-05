import pandas as pd

def load_cgm_data(file):
    df = pd.read_csv(file)
    return df

def load_event_data(file):
    df = pd.read_csv(file)
    return df

# ... 其他必要的函數 ...