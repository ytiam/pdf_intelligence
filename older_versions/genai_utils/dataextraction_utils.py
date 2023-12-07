import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from nltk.tokenize import sent_tokenize

class pdf_extraction_utils():
    
    def __init__(self,pdf_path):
        self.pdf_data_path = pdf_path
        
    def read_pdf(self):
        '''
        Read PDF using Langchain Unstructured PDF Loader api
        '''
        pdf_loader = UnstructuredPDFLoader(self.pdf_data_path, mode="elements")
        documents = pdf_loader.load()
        return documents
    
    def extract_data(self, extraction_category: str = 'NarrativeText'):
        '''
        Extract the raw texts of different categories from the passed pdf
        
        Args:
        extraction_category - the text class which needs to be considered by Langchain while extracting data. Default is NarrativeText
        
        Out:
        df - pandas dataframe. All pagewise textual data of given class extracted and stored in dataframe format
        '''
        documents = self.read_pdf()
        
        all_cats = []
        for doc in documents:
            all_cats.append(doc.metadata['category'])
        
        lis = []
        for i, doc in enumerate(documents):
            if doc.metadata['category'] == extraction_category: 
                lis.append([doc.metadata['page_number'],doc.page_content])
        
        
        check = 1
        temp = []
        temp_data_list = []
        for item in lis:
            if item[0] == check:
                temp.append(item[1])
            else:
                temp_data_list.append([check,"\n\n".join(temp)])
                check += 1
                temp = []
                temp.append(item[1])
                
        df = pd.DataFrame(temp_data_list,columns=["page","raw_texts_"+extraction_category])
        return df
    

# Module to extract multi pdfs, in a single function call
class MultiPDFExtraction():
    
    def __init__(self,pdf_dir: str):
        self.path = pdf_dir
        all_pdfs = os.listdir(self.path)
        all_pdfs = [i for i in all_pdfs if i.endswith('.pdf')]
        self.pdf_id_name_dict = {i:all_pdfs[i] for i in range(0,len(all_pdfs))}
        
    def process_extract_pdfs(self):
        '''
        Function to extract pagewise textual data of a given class (langchain class) for multi pdfs
        '''
        full_df = pd.DataFrame()
        for id_, pdfs in self.pdf_id_name_dict.items():
            try:
                pdf_data = pdf_extraction_utils(self.path + pdfs)
                df = pdf_data.extract_data()
                df["doc_id"] = id_ 
                full_df = pd.concat([full_df,df])
            except:
                continue

        return full_df
    
    def groupby_split_sections_from_text(self, groupd_data):
        sections = groupd_data['sections'].explode()
        sections = sections.reset_index()
        return sections
    
    def detect_single_multi_sentence(self, section: str) -> bool:
        '''
        Function to detect single/multi line presence in an extracted textual block
        
        Args:
        section - textual section. string type.
        
        Out:
        True if multi-line else False
        '''
        list_of_sentences = sent_tokenize(section)
        if len(list_of_sentences) > 1:
            return True
        else:
            return False
    
    def process_extract_sections_pdfs(self, section_spliter: str = "\n\n", extraction_type: str = Union["all","infer"]):
        '''
        Function to extract textual data section wise per page wise, depending on some given condition for extraction type
        
        Args:
        section_spliter - section identifier in textual blocks. default is "\n\n"
        extraction_type - If "all", all the sectional data will be considered. If "infer", section filteration logic will be applied
        and non-eligible sections will be excluded
        
        Out:
        Pandas Dataframe with pagewise sectional data extracted
        
        '''
        text_df = self.process_extract_pdfs()
        text_df['sections'] = text_df['raw_texts_NarrativeText'].str.split(section_spliter)
        
        subset_df = text_df[['doc_id','page','sections']]
        sub_df = subset_df.groupby(['doc_id','page'])
        
        sub_df_section = sub_df.apply(lambda x: self.groupby_split_sections_from_text(x)).reset_index()
        sub_df_section.drop(['index'],axis=1,inplace=True)
        sub_df_section = sub_df_section.rename(columns={'level_2':'section_id'})
        
        if extraction_type == 'infer':
            sub_df_section['is_section'] = sub_df_section['sections'].apply(lambda x: self.detect_single_multi_sentence(x))
            sub_df_section = sub_df_section[sub_df_section.is_section]
            sub_df_section = sub_df_section.reset_index(drop=True)
            sub_df_section = sub_df_section.drop(['is_section'],axis=1)
        return sub_df_section