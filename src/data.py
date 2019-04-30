import preprocess

class Data:
    def __init__(self,n_for_1):
        """
            for train and test data, dataset is in form like [[谜面，答案，地信],]
            for valid data, dataset is [[谜面，谜底，选项],]

            variables begin with `raw` are prepared for calculating valid data.
        """
        # make train, valid, test
        self.n_for_1 = n_for_1
        raw_riddles = preprocess.read_n_for_1_dataset(n_for_1) #[[谜面，谜底，选项],]
        riddles = self.construct_classify_riddle_set(raw_riddles) #[[谜面，答案，地信],]
        one_tenth = len(riddles) // 10
        raw_one_tenth = len(raw_riddles) // 10
        train = riddles[0:one_tenth * 8]
        test = riddles[one_tenth * 8 : one_tenth * 9]
        valid = raw_riddles[raw_one_tenth * 9 : raw_one_tenth * 10]

        # get voc
        self.voc = preprocess.construct_train_data_corpus_vocabulary_dictionary(begin = 0, end = one_tenth * 8)
        self.voc_size = len(self.voc) + 1 # two more place, one for padding, one for unseen
        UNSEEN = self.voc_size + 1

        #indexize
        def indexizer(riddle):
            ret = list()
            ret.append( [self.voc[word] if word in self.voc else UNSEEN for word in riddle[0] ] )
            ret.append( self.voc[riddle[1]] if riddle[1] in self.voc else UNSEEN )
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

    #
    #   @ret: [ riddle, answer, answer with options ], dataset size
    #
    def get_train_data(self):
        return [ [riddle[i] for riddle in self.train] for i in range(3) ],len(self.train)

    def get_valid_data(self):
        return [ [riddle[i] for riddle in self.valid] for i in range(3) ],len(self.valid)

    def get_test_data(self):
        return [ [riddle[i] for riddle in self.test] for i in range(3) ],len(self.test)