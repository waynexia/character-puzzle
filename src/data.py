import preprocess

class Data:
    def __init__(self,n_for_1):
        # get voc
        self.n_for_1 = n_for_1
        self.voc = preprocess.construct_corpus_vocabulary_dictionary()
        self.voc_size = len(self.voc)

        # make train, valid, test
        riddles = preprocess.read_n_for_1_dataset(n_for_1) #[[谜面，谜底，选项],]
        riddles = self.construct_classify_riddle_set(riddles) #[[谜面，答案，地信],]
        num_riddle = len(riddles)
        one_tenth = num_riddle // 10
        train = riddles[0:one_tenth * 8]
        valid = riddles[one_tenth * 8 : one_tenth * 9]
        test = riddles[one_tenth * 9 : one_tenth * 10]

        #indexize
        def indexizer(riddle):
            ret = list()
            ret.append( [self.voc[word] for word in riddle[0]] )
            ret.append( self.voc[riddle[1]] )
            ret.append( riddle[2] )
            return ret

        self.train = [indexizer(row) for row in train]
        self.valid = [indexizer(row) for row in valid]
        self.test = [indexizer(row) for row in test]
        
    def construct_classify_riddle_set(self,riddles):
        #
        # @input:  [[谜面，谜底，选项],]
        # @output: [[谜面，答案，地信],]
        #
        ret = list()
        for riddle in riddles:
            for opt in riddle[2]:
                #item = list()
                #item.append(riddle[0]) #谜面
                #item.append(opt) #答案
                #item.append(opt == riddle[1]) #地信
                ret.append([riddle[0],opt,opt == riddle[1]])
        return ret

    def get_voc_dict(self):
        return self.voc
    def get_voc_size(self):
        return self.voc_size

    ####################
    #
    # ret: [ riddle, answer, answer with options ], dataset size
    #
    ####################
    def get_train_data(self):
        return [ [riddle[i] for riddle in self.train] for i in range(3) ],len(self.train)

    def get_valid_data(self):
        return [ [riddle[i] for riddle in self.valid] for i in range(3) ],len(self.valid)

    def get_test_data(self):
        return [ [riddle[i] for riddle in self.test] for i in range(3) ],len(self.test)