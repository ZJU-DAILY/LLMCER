
import pandas as pd

def get_id_column(df):
    """
    Finds the ID column in the dataframe case-insensitively.
    Returns the column name if found, else None.
    Prioritizes 'ID', then 'id', then 'Id', etc.
    """
    if 'ID' in df.columns:
        return 'ID'
    if 'id' in df.columns:
        return 'id'
    
    # Case-insensitive search
    for col in df.columns:
        if str(col).lower() == 'id':
            return col
    return None
