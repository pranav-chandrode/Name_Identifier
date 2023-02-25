# defining Helper functions
import sys
import io
import os
import string
import glob
import unicodedata

import torch
import random

ALL_LETTERS = string.ascii_letters + " ,.;'"
# print(letters)
N_LETTERS = len(ALL_LETTERS)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )


def load_data():
    def find_files(path):
        # print(1)
        return glob.glob(path)

    all_categories = []
    category_lines = {}

    def read_lines(filename):
        lines = io.open(filename ,encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]


    for filename in find_files('data/names/*.txt'):
        
        kk = os.path.basename(filename)
        k2 = kk.split('.')
        all_categories.append(k2[0])

        lines = read_lines(filename)
        category_lines[k2[0]] = lines

    
    return category_lines,all_categories

def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

def letter_to_tensor(letter):
    tensor  = torch.zeros(1,N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line) ,1,N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1

    return tensor

def random_training_example(category_lines, all_categories):
    
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.zeros(1,18)
    category_tensor[0][all_categories.index(category)] = 1
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

