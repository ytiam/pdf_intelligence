import dataiku
import pandas as pd
import pickle
from pathlib import Path
import os

def dss_read_csv_from_folder(folder,file_name,sep=','):
    folder_path = dataiku.Folder(folder).get_path()
    path_of_csv = os.path.join(folder_path, file_name) 
    embed_data = pd.read_csv(path_of_csv,sep=sep)
    return embed_data

def dss_read_csv_from_folder_as_stream(folder,file_name,sep=','):
    fold = dataiku.Folder(folder)
    with fold.get_download_stream(file_name) as f: 
        embed_data = pd.read_csv(f,sep=sep)
    return embed_data

def dss_read_pickle_from_folder(folder,file_name):
    fold = dataiku.Folder(folder)
    with fold.get_download_stream(file_name) as f:
        data = pickle.load(f)
    return data

def dss_write_csv_to_folder(data,folder,file_name,sep=","):
    fold = dataiku.Folder(folder)
    with fold.get_writer(file_name) as w:
        w.write(data.to_csv(index=False,sep=sep).encode("utf-8"))
             
def dss_write_pickle_to_folder(data,folder,file_name):
    fold = dataiku.Folder(folder)
    with fold.get_writer(file_name) as w:
        w.write(pickle.dumps(data))