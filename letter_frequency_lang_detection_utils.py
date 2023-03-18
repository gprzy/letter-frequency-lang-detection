import re
import string
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from unidecode import unidecode
from sklearn.metrics import mean_squared_error


def get_percentage(string):
    char_list = [char for char in string if char.isnumeric() or char == '.']

    if char_list:
        number = ''
        for char in char_list:
            number += char
        return float(number) / 100
    else:
        return np.nan
    

def read_raw_book(path):
    reader = PdfReader(path)

    book_raw = ''
    for i in range(len(reader.pages)):
        page = reader.pages[i] 
        page_text = page.extract_text()
        book_raw += page_text   

    return book_raw


# characters cleaning

def clean_raw_book(raw_book):
    book = raw_book.lower()
    book = [char for char in book if char.isalpha()]

    book_text = ''
    for char in book:
        book_text += char

    book_text = re.sub(' +', '', book_text)
    book_text = unidecode(book_text)
    return book_text


def count_chars(book):
    characters_freq = {char: 0 for char in string.ascii_lowercase}

    for char in book:
        if char in list(characters_freq.keys()):
            characters_freq[char] += 1

    characters_freq = {k: [v] for k, v in characters_freq.items()}
    return characters_freq


def get_freq_df(freq_dict, book):
    df_freq = pd.DataFrame(freq_dict)

    total_chars = len(book)

    df_freq = df_freq / total_chars
    return df_freq


def mse_predict(sample, df_freq, X):
    languages = df_freq['language'].values

    mse = []
    for i in range(len(languages)):
        mse.append(mean_squared_error(X[i], sample[0]))

    index = np.argmin(mse)
    return languages[index]
