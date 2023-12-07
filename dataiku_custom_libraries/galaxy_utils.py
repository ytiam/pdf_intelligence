import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
import dataikuapi
from dataiku_utils import *
# Used to create the dense document vectors.
from sentence_transformers import SentenceTransformer
import faiss
# Used to create and store the Faiss index.
import numpy as np
import pickle
from pathlib import Path
import os
import string
from flask import request
import pytesseract
from pdf2image import convert_from_path
from pdf2image import convert_from_bytes
import io
import gc
import re
import random
import cv2
import spacy
nlp = spacy.load('en_core_web_sm')
        
def text_filter(x,table):
    words = x.split()
    stripped = [w.translate(table) for w in words]
    c = ' '.join(stripped)
    return c

def vector_search(query, num_results=10, index=None, model=None):
    index = index
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I

def id2details(query, df ,column=[],num_results=10,index=None,model=None):
    """Returns the paper titles based on the paper index."""
    D, I = vector_search([query],num_results=num_results,index=index,model=model)
    return [df[df.id_process == idx][column].values.tolist()[0] for idx in I[0]]

def text_preprocess(text):
    return text.lower().replace('\n', ' ').replace('\t', ' ').replace('\x0c',' ')

def generate_id(i,k,len_):
    page_no_max_len = len(str(len_))
    cur_page_no_len = len(str(k))
    req_zeros = page_no_max_len - cur_page_no_len
    id_ = int(i.replace('_','')+'0'*req_zeros+str(k))
    return id_

def sentence_detect(x):
    doc = nlp(x)
    temp_lis = []
    for i in doc.sents:
        temp_lis.append(" ".join(str(i).split()))
    return temp_lis

def new_id_generate(file_name):
    seed = int(file_name,36)
    random.seed(seed)
    new_id = random.randint(10**8, 10**9)
    return new_id

def validate_generate_doc_id(file_name_tuple):
    k = file_name_tuple[0]
    v = file_name_tuple[1]
    id_ = new_id_generate(v)
    return id_

def process_file(file,wcdf_folder_handle):
    file_name = file.split('.pdf')[0].replace('/','')
    f = wcdf_folder_handle.get_file(file).content
    images = convert_from_bytes(f,150)
    file_contents = []
    for i,im in enumerate(images):
        img_ = np.array(im)
        fileReader = pytesseract.image_to_string(img_)
        file_contents.append(fileReader)
    file_content = '</np>'.join(file_contents)
    return file_content

def run_process():
    ids = dss_read_pickle_from_folder("misc","doc_ids.pickle")
    fold = dataiku.Folder("new_data")
    folder_id = fold.get_id()
    client = dataiku.api_client()
    project = dataikuapi.dss.project.DSSProject(client,'GALAXY_1')
    wcdf_folder_handle = dataikuapi.dss.managedfolder.DSSManagedFolder(client,'GALAXY_1',folder_id)
    ff = fold.get_path_details()
    file_list = ff['children']
    files = []
    for i in range(len(file_list)):
        files.append(file_list[i]['fullPath'])
    temp = []
    id_temp = []
    for file in files:
        present_file_name = file.split('.pdf')[0].replace('/','')
        modified_file_name = present_file_name.replace('-','')
        modified_file_name = modified_file_name.replace('.','')
        modified_file_name = ''.join(modified_file_name.split())
        modified_file_name = modified_file_name.lower()
        modified_file_name = re.sub('[^a-zA-Z0-9 \n\.]', '', modified_file_name)
        print(present_file_name,modified_file_name)
        file_id = validate_generate_doc_id((present_file_name,modified_file_name))
        temp.append([str(file_id),process_file(file,wcdf_folder_handle)])
        ### ID mapping table update
        ids = ids.append({"doc":present_file_name,"doc_id_parsed":str(file_id)},ignore_index=True)
        ids = ids.drop_duplicates()
        dss_write_pickle_to_folder(ids,"misc","doc_ids.pickle")
    df_temp = pd.DataFrame(temp, columns=['doc','text'])
    df_temp['text'] = df_temp['text'].apply(lambda x: text_preprocess(x))
    dss_write_csv_to_folder(df_temp,"new_text_extracted_data_table","failure_analysis_reports.csv")
    return df_temp

def preprocess_text(my_dataset):
    my_dataset = my_dataset[~my_dataset.text.isnull()]
    page_len = []
    for i in my_dataset.doc.unique():
        k = my_dataset.loc[(my_dataset['doc']==i),'text'].values[0]
        temp_text = k.split('</np>')
        len_ = len(temp_text)
        page_len.append(len_)
    expand_data = []
    max_page_len = max(page_len)
    for i in my_dataset.doc.unique():
        k = my_dataset.loc[(my_dataset['doc']==i),'text'].values[0]
        temp_text = k.split('</np>')
        for k,j in enumerate(temp_text):
            expand_data.append([i,j,generate_id(i,k,max_page_len)])
    df = pd.DataFrame(expand_data,columns=['doc_id','text','id'])
    df['processed_text'] = df['text'].apply(lambda x: sentence_detect(x))
    df_sub = df[['id','processed_text']]
    rows = []
    _ = df_sub.apply(lambda row: [rows.append([row['id'], i, nn]) 
                             for i, nn in enumerate(row.processed_text)], axis=1)
    df_new = pd.DataFrame(rows, columns=['id','sent_id','sentences'])
    df_sentences = pd.merge(df,df_new,on='id',how='left')
    df_sentences = df_sentences.dropna(subset=['sent_id'])
    df_sentences['sent_id'] = df_sentences['sent_id'].astype('int')
    df_sentences['id_process'] = df_sentences[['id','sent_id']].apply(lambda x: int(str(x[0])+str(x[1])),axis=1)
    table = str.maketrans('', '', string.punctuation)
    df_sentences['sentences'] = df_sentences['sentences'].apply(lambda x: text_filter(x,table))
    df_sentences = df_sentences[~(df_sentences['sentences']=='')]
    dss_write_pickle_to_folder(df_sentences,"new_preprocessed_text_data","pdf_text_processed.pickle")
    return df_sentences

def process_embed(model,df):
    df['encoded'] = df['sentences'].apply(lambda x: model.encode(x))
    df = df[['doc_id','id','sent_id','sentences','id_process','encoded']]
    dss_write_pickle_to_folder(df,"new_encoded_data","encoded_file.pickle")
    return df

def update_index(df):
    index = dss_read_pickle_from_folder("embedded_index","index.pickle")
    embeddings = df.encoded.to_list()
    embeddings = np.array(embeddings)
    embeddings = np.array([embedding for embedding in embeddings]).astype("float32")
    index.add_with_ids(embeddings, df.id_process.values)
    dss_write_pickle_to_folder(index,"updated_embedded_index","index.pickle")
    return index

def update_encoded_data(df,df_new):
    old_encoded_data = df
    new_encoded_data = df_new
    new_encoded_data = pd.concat([old_encoded_data,new_encoded_data])
    dss_write_pickle_to_folder(new_encoded_data,"updated_encoded_data","encoded_file.pickle")
    return new_encoded_data

def sync_folders():
    client = dataiku.api_client()
    project = client.get_project('GALAXY_1')
    fold1 = dataiku.Folder("updated_encoded_data")
    folder1_id = fold1.get_id()
    source_folder = project.get_managed_folder(folder1_id)
    fold2 = dataiku.Folder("prod_encoded_data")
    folder2_id = fold2.get_id()
    target_folder = project.get_managed_folder(folder2_id)
    future = source_folder.copy_to(target_folder)
    future.wait_for_result()
    fold1 = dataiku.Folder("updated_embedded_index")
    folder1_id = fold1.get_id()
    source_folder = project.get_managed_folder(folder1_id)
    fold2 = dataiku.Folder("prod_embedded_index")
    folder2_id = fold2.get_id()
    target_folder = project.get_managed_folder(folder2_id)
    future = source_folder.copy_to(target_folder)
    future.wait_for_result()