import jieba.posseg as pseg
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import math
from langchain.text_splitter import CharacterTextSplitter
import os
# Interpunction_array = ['.', ';', '!', '?', '。', '；', '？', '！']

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200, # 200
    chunk_overlap=0,
    length_function=len
)

def read_data(path):
    text_array = []
    with open(path, "r") as fin:
        for line in fin:
            if len(line.strip()) > 0:
                text_array.append(line)
    return text_array

def create_qa_text(question: str, answer: str):
    # return "Question: " + question + "\nFactual answer:" + answer
    return {'question': question.strip(), 'answer': answer.strip()}

def split_text(text):
    return text_splitter.split_text(text)

def load_excel(file_path):
    # print('..............load_excel:', file_path)
    order_dict = pd.read_excel(file_path, sheet_name=None, index_col=None, header=None)
    chunks = []
    for sheet_name, df in order_dict.items():
        # print(sheet_name)
        question_index = -1
        answer_index = -1
        row_index = 0
        for row in df.values:
            row_data_list = []
            question = None
            answer = None
            for col_index in range(len(row)):
                data = row[col_index]
                if not (type(data) is float and math.isnan(data)):
                    if col_index == question_index:
                        question = data
                    elif col_index == answer_index:
                        answer = data
                    if row_index == 0:
                        if data.strip() == '标准问':
                            question_index = col_index
                        elif data.strip() == '回答':
                            answer_index = col_index
                    row_data_list.append(data)
            if question_index == -1 and answer_index == -1:
                for data in row_data_list:
                    chunks.append(data)
            elif question is not None and answer is not None:
                # chunks.append("当提问：\"" + question + "\"时，请用中文回答：\"" + answer + "\"")
                chunks.append(create_qa_text(question, answer))
                # chunks.append(question)
            row_index += 1
    return chunks

def load_text_list_for_dir(dir_path):
    if os.path.isdir(dir_path):
        data_list = []
        files = os.listdir(dir_path)
        if not dir_path.endswith('/'):
            dir_path += '/'
        for file in files:
            # print(dir_path + file)
            d_list = load_text_list(dir_path + file)
            data_list.extend(d_list)
            # print('.........data_list len:', len(data_list))
        return data_list
    else:
        return load_text_list(dir_path)

def load_text_list(file_path):
    if file_path.lower().endswith('.txt'):
        text_array = read_data(file_path)
        return text_array
    elif file_path.lower().endswith('.pdf'):
        pdf_reader = PdfReader(file_path)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text_splitter.split_text(text)
    elif file_path.lower().endswith('.docx') or file_path.lower().endswith('.doc'):
        document = Document(file_path)
        text = ''
        for paragraph in document.paragraphs:
            text += paragraph.text + '\r\n'
        return text_splitter.split_text(text)
    elif file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls'):
        return load_excel(file_path)
    return []

def filter(text):
    if len(text) == 0:
        return text
    if not contains_chinese_chars(text):
        if 'i don\'t know.' == text.lower():
            return ''
        else:
            return text
    data_list = pseg.cut(text)
    # print('...........data_list:', data_list)
    word_list = []
    index = 0
    interpunction_index = -1
    for word, flag in data_list:
        word_list.append(word)
        if flag == 'x':
            interpunction_index = index
        index += 1
    # print('...............interpunction_index:', interpunction_index)
    if interpunction_index > 0:
        word_list = word_list[0: interpunction_index + 1]
    return ''.join(word_list)

def contains_chinese_chars(s: str) -> bool:
    for c in s:
        if '\u4e00' <= c <= '\u9fa5':
            return True
    return False

if __name__ == '__main__':
    # print(filter(u'习近平发表机场书面讲话，代表中国政府和中国人民，向南非政府和南非人民致以诚挚问候和良好祝愿。习近平强调，今年是中南建交25周年，中南全面战略伙伴关系步入崭新阶段。中南关系的巩固和发展，不仅造福两国人民，也为变乱交织的世界注入更多稳定性。我相'))
    # print(load_excel('../data/data.xlsx'))
    load_text_list_for_dir('../data/data2')