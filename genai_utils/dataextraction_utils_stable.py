from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from nltk.tokenize import sent_tokenize
import pandas as pd
from typing import Union
import pdfplumber
import fitz
import glob
import sys
import os
from glob import glob
import re
from collections import OrderedDict
import requests
import pickle


class pdf_data_extraction_utils():

    def __init__(self, pdf_path):
        self.set_header_type = None
        self.pdf_data_path = pdf_path
        self.doc = self.read_pdf()

    def read_pdf(self):
        '''
        Read PDF using Langchain Unstructured PDF Loader api
        '''
        return fitz.open(self.pdf_data_path)

    def save_page_images(self, image_fold_path: str = None):
        '''
        To save each individual pages as images for a PDF
        
        Args:
        image_fold_path - the folder path where the images will be stored
        
        Output:
        fold_path - images are saved and the path is returned back for further use
        '''
        zoom_x = 1.0  # horizontal zoom
        zoom_y = 1.0  # vertical zoom
        mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

        if image_fold_path is not None:
            image_path = image_fold_path
        else:
            image_path = os.path.dirname(
                os.path.abspath(
                    self.pdf_data_path)) + '/out'

        fold = os.path.basename(self.pdf_data_path).replace('.pdf', '')

        fold_path = os.path.join(image_path, fold)

        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        if len(glob(fold_path + '/*.jpg')) == 0:
            print("Converting Pages in Images...")
            for page in self.doc:  # iterate through the pages
                pix = page.get_pixmap(matrix=mat)  # render page to an image
                pix.save("%s/page-%i.jpg" % (fold_path, page.number))
        return fold_path

    def remove_unicode(self, x:str):
        '''
        Remove unicode characters from a string
        '''
        return x.encode("ascii", "ignore").decode()

    def get_table_of_content(self):
        '''
        Retrive table of content from Digital PDF using PyMuPDF functionality
        '''
        return self.doc.get_toc()

    def get_page_content(self, page_no):
        '''
        Get a PDF page text content for a given page number
        
        Arg:
        page_no - page number
        
        Output:
        blocks - list of text lines for a given page
        '''
        pg = self.doc.load_page(page_no)
        paras = pg.get_text('blocks')

        blocks = []
        for para in paras:
            para_content = re.sub(
                ' +',
                ' ',
                para[4].replace(
                    '\n',
                    ' ').strip())
            blocks.append(para_content)
        return blocks

    def section_name_splitter(self, x):
        '''
        Split a string (section header text) when and as in required
        
        Arg:
        x - retrived header string
        
        Output:
        splitted header list
        '''
        spliter = re.findall("[0-9].1.", x)[0]
        hh = x.split(f' {spliter} ')
        return [hh[0], spliter + ' ' + hh[1]]

    def section_name_merger(self, x: list):
        '''
        Maerge a string (section header text) when and as in required
        
        Arg:
        x - header list
        
        Output:
        Merged header string
        '''
        merged_string = f"{x[0]} {x[1]}"
        return merged_string

    def extract_page_data(self):
        '''
        Extract the raw texts of different categories from the passed pdf
        '''
        documents = self.doc

        page_blocks = {}
        for pg_no in range(documents.page_count):
            pg_content = documents.load_page(pg_no)
            paras = pg_content.get_text('blocks')
            blocks = []
            for para in paras:
                blocks.append(para[4].replace('\n', ' ').strip())

            page_blocks[pg_no] = blocks
        return page_blocks
    
    def get_table_of_content_ocr(
            self,
            image_fold_path=None,
            header_type='caps'):
        '''
        Retrive table of content using OCR method (LayoutParser DeepLearning Based Method)
        
        Args:
        image_fold_path - Default None. If Default, then the image saving path will be auto assigned and images will be generated
        and stored accordingly
        header_type - ['caps' or 'all']. Default is 'caps'. If default, only headers with all capital letter will be returend, else
        all headers identified will be returned
        
        Output:
        pg_headers - list of all the headers and extracted
        
        '''
        img_fold_path = self.save_page_images(image_fold_path=image_fold_path)

        if os.path.exists(f"{img_fold_path}/header.h"):
            with open(f"{img_fold_path}/header.h", "rb") as f:
                pg_headers = pickle.load(f)
        else:
            pg_headers = []
            for i, pg in enumerate(self.doc):
                filename = f"{img_fold_path}/page-{str(i)}.jpg"
                files = {'my_file': (filename, open(filename, 'rb'))}

                response = requests.post(
                    'http://20.83.24.160:8889/getLayout',
                    files=files)
                headers = response.json()['header']

                headers = [i for i in headers if i != '']

                if self.set_header_type is not None:
                    header_type = self.set_header_type

                if header_type == 'caps':
                    headers = [j for j in headers if j.isupper()]

                if len(headers) != 0:
                    pg_content = self.get_page_content(i)

                    # pg_content = [' '.join(pg_content)]

                    line_headers = OrderedDict()
                    for j, line in enumerate(pg_content):
                        temp_line_header = []
                        for h in headers:
                            if h in line:
                                temp_line_header.append(h)
                        line_headers[line] = temp_line_header

                    line_headers = {
                        k: v for k, v in line_headers.items() if len(v) != 0}

                    headers_sorted = []
                    for k, v in line_headers.items():
                        if len(v) > 1:
                            v.sort(key=len, reverse=True)
                            for it_ in v:
                                k = k.replace(it_, '_'.join(it_.split()))

                            k_split = k.split(" ")

                            index_dict = {}

                            for it_ in v:
                                try:
                                    index_dict[it_] = k_split.index(
                                        '_'.join(it_.split()))
                                except BaseException:
                                    continue

                            index_dict_key_sorted = [k for k, v in sorted(
                                index_dict.items(), key=lambda item: item[1])]
                            headers_sorted.extend(index_dict_key_sorted)
                        else:
                            headers_sorted.append(v[0])
                    for h in headers_sorted:
                        pg_headers.append([1, h, i + 1])

            with open(f"{img_fold_path}/header.h", "wb") as f:
                pickle.dump(pg_headers, f)
        return pg_headers

    def get_content_structure(self):
        '''
        Get headers configured and modified from the list of headers extracted, to identify and extract the sectional data
        from the raw page texts
        
        Output:
        process_flag - either headers extracted through 'digital'/'ocr' method
        temp - list of lists. each list item is [section_header, page_no, header_level (root/child)]
        
        '''
        temp = []
        sec_count = 0
        sub_sec = 0

        if self.extraction_mode == "auto":
            l = self.get_table_of_content()
            process_flag = "digital"
            if len(l) == 0:
                l = self.get_table_of_content_ocr()
                process_flag = "ocr"
        elif self.extraction_mode == 'digital':
            l = self.get_table_of_content()
            process_flag = "digital"
        elif self.extraction_mode == 'ocr':
            l = self.get_table_of_content_ocr()
            process_flag = "ocr"

        for j in range(0, len(l)):
            tag = 'root'
            i = l[j]
            level, section_name, page = i[0], i[1].strip(), i[2]
            if level == 1:
                root_section_name = section_name
                sec_count = sec_count + 1
                sub_sec = 0
                try:
                    if l[j + 1][0] > 1:
                        tag = "root_with_child"
                except BaseException:
                    pass
            if level > 1:
                tag = 'child'
                sub_sec += 1
                section_text_name = f'{sec_count}.{sub_sec}. {section_name}'
            elif section_name == "References":
                section_text_name = f'{section_name}'
            else:
                section_text_name = f'{sec_count}. {section_name}'
            temp.append([section_text_name, page, tag])
        return process_flag, temp

    def get_content_structure_filtered(self):
        '''
        Do some filteration on the item set returned by get_content_structure function
        
        '''
        _, all_content_struc = self.get_content_structure()
        filtered_content_struc = [[i[0].encode("ascii", "ignore").decode(
        ), i[1]] for i in all_content_struc]  # if i[2]!= 'root_with_child']
        # filtered_content_struc = [[re.sub(r'[^\x00-\x7F]+',' ', i[0]), i[1]] for i in all_content_struc]
        filtered_content_struc.append(['DUMMY', filtered_content_struc[-1][1]])
        return _, filtered_content_struc

    def get_content_section_wise(
            self,
            extraction_mode='auto',
            set_header_type=None):
        '''
        Main function to extract the section wise textual content with the help of above defined functional components
        
        Args:
        extraction_mode - Default is 'auto'. Acceptable values are ['auto', 'digital', 'ocr']. these are different modes through
        which the header and sectional data can be extracted
        set_header_type - Either 'caps' or 'all'. Similar utilization like get_table_of_content_ocr function
        
        Output:
        dic - Dictionary Type. {header: text content under the header}
        
        '''

        if set_header_type is None:
            self.set_header_type = 'all'
        else:
            self.set_header_type = set_header_type

        self.extraction_mode = extraction_mode

        process_flag, ll = self.get_content_structure_filtered()

        print(ll)
        print(f"\n\nProcessing through : {process_flag}")
        if process_flag == "digital":
            all_sec_content = []
            j = 0
            sec_content = ''
            breaker = 3
            dic = {}

            try:
                while j < len(ll) - 1:
                    sec_start_page_no, start_sec_name = ll[j][1], ll[j][0]
                    sec_end_page_no, end_sec_name = ll[j + 1][1], ll[j + 1][0]
                    flag = False
                    for i in range(sec_start_page_no, sec_end_page_no + 1):
                        pg = self.doc.load_page(i - 1)
                        paras = pg.get_text('blocks')

                        blocks = []
                        for para in paras:
                            blocks.append(
                                para[4].replace(
                                    '\n', ' ').strip().encode(
                                    "ascii", "ignore").decode())

                        if i != sec_end_page_no:
                            sec_content += ' '.join(blocks)
                        else:
                            try:
                                next_section_header_index = blocks.index(
                                    end_sec_name)

                                till_pg_section = blocks[:next_section_header_index]
                                sec_content += ' '.join(till_pg_section)

                                next_sec_start_section = blocks[next_section_header_index:]
                                next_sec_start_content = ' '.join(
                                    next_sec_start_section)
                            except Exception as e:
                                # print(e)
                                # print(blocks, " ---- ", start_sec_name, " --- ",end_sec_name,">>>>>>>>>>>>>>>>> \n\n")
                                if end_sec_name != "References":
                                    new_ll = ll[:j + 1]
                                    n_1 = ll[j + 1][0]
                                    n_2 = ll[j + 2][0]
                                    merged_section_name = pdf.section_name_merger([
                                                                                  n_1, n_2])
                                    new_ll.append(
                                        [merged_section_name, sec_end_page_no])
                                    new_ll.extend(ll[j + 3:])
                                    ll = new_ll
                                else:
                                    ll[-2][1] += 1
                                flag = True
                                break

                    if not flag:
                        dic[start_sec_name] = sec_content
                        # all_sec_content.append([start_sec_name, sec_content])
                        sec_content = next_sec_start_content
                        j += 1
                    else:
                        sec_content = ''
            except Exception as e:
                # print(e)
                pass
            return dic
        else:
            all_sec_content = []
            j = 0
            full_content = ''

            for pg_no in range(self.doc.page_count):
                full_content += ' '.join(self.get_page_content(pg_no))

            dic = {}
            for i, sec_pg in enumerate(ll):
                header = sec_pg[0]
                temp = full_content.split(header)
                if len(temp) == 1:
                    try:
                        number = re.findall("[0-9]+. ", header)[0]
                        # print(header,": ",number,"#####\n\n")
                        header_ = header.replace(number, '')
                        header_ = re.sub(' +', ' ', header_)
                    except Exception as e:
                        header_ = header

                    try:
                        temp = full_content.split(header_)
                        first_por = temp[0]
                        second_por = temp[1]
                    except Exception as e:
                        # print(e)
                        pass
                else:
                    first_por = temp[0]
                    second_por = temp[1]

                if i > 0:
                    dic[prev_header_mem] = first_por
                prev_header_mem = header
                full_content = second_por

            return dic
