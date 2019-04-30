import random
import os
import csv

# data dirctory
data_dir = "../data/"

# construct corpus voc dictionary
def construct_train_data_corpus_vocabulary_dictionary(begin,end):
    corpus_file_name = "riddle.csv"
    voc = dict()
    cnt = 0
    line_cnt = begin
    with open( os.path.join(data_dir,corpus_file_name), "r") as f:
        for line in f:
            if line_cnt >= end:
                break
            for word in line:
                if word == ',':
                    continue
                if word not in voc:
                    voc[word] = cnt
                    cnt += 1
            line_cnt += 1
    return voc

# construct answer voc dictionary
def construct_answer_vocabulary_dictionary():
    ans_file_name = "ans_cnt.csv"
    ans_voc_dict = dict()
    ans_voc_list = list()
    cnt = 0
    with open( os.path.join(data_dir,ans_file_name), "r") as f:
        for line in f:
            ans_word = line[0]
            if ans_word not in ans_voc_dict:
                ans_voc_dict[ans_word] = cnt
                ans_voc_list.append(ans_word)
                cnt += 1
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


if __name__ == "__main__":
    _,l = construct_answer_vocabulary_dictionary()
    construct_n_for_1_dataset(2,l)
    construct_n_for_1_dataset(5,l)
    construct_n_for_1_dataset(10,l)

    
            






