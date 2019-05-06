import random
import os
import copy
import csv
import pprint
import torch

# data dirctory
data_dir = "../data/"

# construct corpus voc dictionary
def construct_train_data_corpus_vocabulary_dictionary(begin,end):
    corpus_file_name = "riddle.csv"
    riddle_voc = dict()
    ans_voc = dict()
    riddle_cnt = 0
    ans_cnt = 0
    line_cnt = begin
    with open( os.path.join(data_dir,corpus_file_name), "r") as f:
        for line in f:
            if line_cnt >= end:
                break
            riddle,ans = line.split(',')
            for word in riddle:
                if word == ',':
                    continue
                if word not in riddle_voc:
                    riddle_voc[word] = riddle_cnt
                    riddle_cnt += 1
            for word in ans:
                if word == ',':
                    continue
                if word not in ans_voc:
                    ans_voc[word] = ans_cnt
                    ans_cnt += 1
            line_cnt += 1
    return riddle_voc,ans_voc

# construct answer voc dictionary
def construct_answer_vocabulary_dictionary():
    ans_file_name = "ans_cnt.csv"
    ans_voc_list = list()
    ans_voc_dict = dict()
    cnt = 0
    with open( os.path.join(data_dir,ans_file_name), "r") as f:
        for line in f:
            ans_word = line[0]
            if ans_word not in ans_voc_dict:
                ans_voc_dict[ans_word] = cnt
                ans_voc_list.append(ans_word)
                cnt += 1
    
    # construct ans word to seq dict
    ans_voc_dict = dict()
    with open( os.path.join(data_dir,"riddle_seq.csv"), "r") as f:
        for line in f:
            _,ans_word,_,ans_seq = line.split(',')
            if ans_word not in ans_voc_dict:
                ans_voc_dict[ans_word] = [digit for digit in ans_seq]


    return ans_voc_dict, ans_voc_list

# construct n for 1 dataset and dirct write into file
def construct_n_for_1_dataset(n,ans_list):
    corpus_file_name = "riddle.csv"
    out_file = str(n) + "_for_1.csv"
    with open( os.path.join(data_dir,corpus_file_name), "r") as f:
        with open( os.path.join(data_dir,out_file) , "w") as out:
            writter = csv.writer(out, delimiter=',')
            reader = csv.reader(f)
            for row in reader:
                sample = ""
                for word in random.sample( [i for i in ans_list if i != row[1]], n-1 ):
                    sample += word
                sample += row[1]
                row.append(sample)
                writter.writerow(row)

def read_n_for_1_dataset(n):
    dataset_file = str(n) + "_for_1.csv"
    try:
        with open( os.path.join(data_dir,dataset_file),'r') as f:
            reader = csv.reader(f)
            ret = [row for row in reader]
    except FileNotFoundError:
        print("file name error, file not exist")
        exit(0)
    return ret

"""
def get_stroke():
    seq_dict = dict()
    with open( os.path.join(data_dir,"riddle_seq.csv"), "r") as f:
        for line in f:
            riddle_words,ans_word,riddle_seqs,ans_seq = line.split(',')
            if ans_word not in seq_dict:
                seq_dict[ans_word] = [digit for digit in ans_seq]
            word_cnt = 0
            for riddle_word in riddle_words:
                if riddle_word == '，' or riddle_word == ' ':
                    continue
                if riddle_word 
                word_cnt += 1
"""
def get_stroke(ans_voc_dict,ans_voc_list,riddle_voc,ans_voc):
    #riddle_voc,ans_voc = construct_train_data_corpus_vocabulary_dictionary(begin,end)
    riddle_unk = len(riddle_voc)
    ans_unk = len(ans_voc)
    print(riddle_unk,ans_unk)
    ret = list()

    with open( os.path.join(data_dir,"riddle_seq.csv"), "r") as f:
        for line in f:
            riddle,ans,r_seq,a_seq = line.split(',')
            new_item = dict()
            new_item['puzzle'] = list()
            for word in riddle:
                if word == ' ':
                    continue
                new_item['puzzle'].append(riddle_voc[word] if word in riddle_voc else riddle_unk)
            new_item['ans'] = ans_voc[ans] if ans in ans_voc else ans_unk
            new_item['label'] = 1
            new_item['stroke_puzzle'] = [[int(digit) for digit in char] for char in r_seq.split(' ')]
            new_item['stroke_ans'] = [int(digit) for digit in a_seq[:-1]]

            if len(new_item['puzzle']) != len(new_item['stroke_puzzle']):
                continue
            ret.append(new_item)
            new_item = copy.deepcopy(new_item)

            new_ans_i = new_item['ans']
            while new_ans_i == new_item['ans']:
                new_ans_c = random.choice(ans_voc_list)
                new_ans_i = ans_voc[new_ans_c] if new_ans_c in ans_voc else ans_unk
            new_item['ans'] = new_ans_i
            new_item['label'] = 0
            new_item['stroke_ans'] = [int(digit) for digit in ans_voc_dict[new_ans_c][:-1]]
            ret.append(new_item)
    return ret


if __name__ == "__main__":
    """ans_voc_dict,l = construct_answer_vocabulary_dictionary()
    construct_n_for_1_dataset(2,l)
    construct_n_for_1_dataset(5,l)
    construct_n_for_1_dataset(10,l)"""
    a,b = construct_answer_vocabulary_dictionary()
    riddle_voc,ans_voc = construct_train_data_corpus_vocabulary_dictionary(0,9544)
    ret = get_stroke(a,b,riddle_voc,ans_voc)
    length_1_10 = len(ret) // 10
    for item in ret:
        if len(item['puzzle']) != len(item['stroke_puzzle']):
            print(item)
    torch.save(ret[0:length_1_10*8],"train_with_stroke.pt")
    torch.save(ret[length_1_10 *8:length_1_10*9],"test_with_stroke.pt")
    

    
            






