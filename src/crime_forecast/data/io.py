def read_csv(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def read_excel(file_path):
    import pandas as pd
    return pd.read_excel(file_path)

def save_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)

def save_to_excel(dataframe, file_path):
    dataframe.to_excel(file_path, index=False)