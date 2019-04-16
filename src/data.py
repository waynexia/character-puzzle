import preprocess

class Data:
    def __init__(self,n_for_1):
        # get voc
        self.n_for_1 = n_for_1
        self.voc = preprocess.construct_corpus_vocabulary_dictionary()
        self.voc_size = len(self.voc)

        # make train, valid, test
        riddles = preprocess.read_n_for_1_dataset(n_for_1)
        num_riddle = len(riddles)
        shuffle(riddles)
        one_tenth = nun_riddle // 10
        train = riddles[0:one_tenth * 8]
        valid = riddles[one_tenth * 8 : one_tenth * 9]
        test = riddles[one_tenth * 9 : one_tenth * 10]

        #indexize
        def indexizer(riddle):
            ret = list()
            ret.append( [voc[word] for word in riddle[0]] )
            ret.append( voc[riddle[1]] )
            ret.append( [voc[word] for word in riddle[0]] )
            return ret
        self.train = [indexizer(row) for row in train]
        self.valid = [indexizer(row) for row in train]
        self.test = [indexizer(row) for row in test]
        


    def get_voc_dict(self):
        return self.voc

    ####################
    #
    # ret: [ riddle, answer, answer with options ], dataset size
    #
    ####################
    def get_train_data(self):
        shuffle(self.train)
        return [ [riddle[i] for riddle in self.train] for i in range(3) ],len(self.train)

    def get_valid_data(self):
        shuffle(self.valid)
        return [ [riddle[i] for riddle in self.valid] for i in range(3) ],len(self.valid)

    def get_test_data(self):
        shuffle(self.test)
        return [ [riddle[i] for riddle in self.test] for i in range(3) ],len(self.test)