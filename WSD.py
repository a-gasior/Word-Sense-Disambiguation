#Andrew Gasiorowski

import re as _re
import pandas as _pd
import random
import sys
from collections import Counter as _Counter
from math import ceil as _ceil
from math import log as _log

#this function takes a file name and returns a list of tuples
#tuple[0] is answer instance
#tuple[1] is word sense
#tuple[2] is context text
def extract_tuples(file_name):
    with open(file_name) as file:
        corpus = _re.findall(r'<answer instance="(.+?)" senseid="(.+?)"/>.+?<context>.(.+?).</context>',file.read(), _re.DOTALL)
    return corpus

#This function takes a list of tuples where the context (tuple[2]) is a single string. It returns a new list where tuple[0&1] are the same but tuple[2] is now a list
def extract_context(corpus):
    enhanced_corpus = []
    word_senses = []
    for instance in corpus:
        #for every training sample
        if instance[1] not in word_senses:
            #if this word sense has not been seen before
            word_senses.append(instance[1])
            #add it to the list of known word senses
        words = _re.findall(r'(?=(?<!<|>)\b)(\w+)(?=(?!>)\b)', instance[2])
        #extract all the words that aren't the word we're trying to disambiguate(or it's tag words)
        new_tuple = (instance[0], instance[1], words)
        enhanced_corpus.append(new_tuple)
    return enhanced_corpus, word_senses

#This function generates the indexes for folds in a top-bottom approach from the input file
def generate_k_fold_validation_indexes(corpus, senses, k):
    num_senses = len(senses)
    num_samples = len(corpus)
    indexes = []
    if num_samples % k != 0:
        #if we cannot evenly divide the training samples
        num_samples_not_last_fold = _ceil(num_samples/k)
        #the equal size folds are of size #samples / #folds
        num_samples_last_fold = num_samples - ((k - 1) * num_samples_not_last_fold)
        #the last fold is all the remaining samples
        this_start = 0
        this_end = num_samples_not_last_fold
        for i in range(0,k-1):
            #for each of the equal sized folds
            indexes.append(list(range(this_start,this_end)))
            #append a list of indexes
            this_start = this_end
            this_end = this_end + num_samples_not_last_fold
            #The above two lines move the indexes
        indexes.append(list(range(this_start, this_start + num_samples_last_fold)))
        #append a final list of indexes of the remaining numbers
    else:
        #if we can evenly divide our training samples
        num_samples_per_fold = _ceil(num_samples/k)
        this_start = 0
        this_end = num_samples_per_fold
        for i in range(0,k):
            #for each of the folds
            indexes.append(list(range(this_start,this_end)))
            #append a list of indexes
            this_start = this_end
            this_end = this_end + num_samples_per_fold
    return indexes

#this function was created as an optimization of the previous index algorithm
#If we have M samples we generate M unique random numbers between 0 : (M-1)
#These numbers are stored in a list. We iterate through the list of rands and generate the indexes of the folds
#It returns a list of lists where each nested list contains the indexes for a fold
def generate_rands(corpus, senses, k):
    num_samples = len(corpus)
    raw_indexes = _pd.Series(random.sample(range(num_samples), num_samples))
    #this generates the random unique numbers
    indexes = []
    this_start = 0
    if num_samples % k == 0:
        #if we can evenly split the number of samples
        num_samples_per_fold = int(num_samples/k)
        #every fold will have the same number of samples
        this_end = num_samples_per_fold
        #this end is used to signify the list index of the end of the current fold
        for i in range(0,k):
            indexes.append(raw_indexes[this_start:this_end])
            #append this list of indexes to the ith fold
            this_start = this_end
            #the start of the next fold is the end of this fold
            this_end = this_end + num_samples_per_fold
            #the end of the next fold is the end of this fold plus the fold size          
    else:
        #if we cannot evenly split the number of samples
        num_samples_not_last_fold = int(_ceil(num_samples/k))
        num_samples_last_fold = int(num_samples - ((k - 1) * num_samples_not_last_fold))
        #the last fold will contain the remainder of examples
        this_end = num_samples_not_last_fold
        for i in range(0,k-1):
            #assign all the evenly sized folds
            indexes.append(raw_indexes[this_start:this_end])
            this_start = this_end
            this_end = this_end + num_samples_not_last_fold
        indexes.append(raw_indexes[this_end - num_samples_not_last_fold:this_end+num_samples_last_fold])
        #assign the remaining examples to the last fold
    return indexes

#this function generates folds with the aim of minimizing the difference between the number of samples per fold accross all folds
def even_folds(corpus, senses, k):
    sense_list_in_order = []
    for sample in corpus:
        sense_list_in_order.append(sample[1])
    num_per_sense = _Counter(sense_list_in_order)
    count_per_fold = {}
    for sense in senses:
        total = num_per_sense[sense]
        not_last_fold = _ceil(total / k)
        total_not_last = (k-1)*not_last_fold
        last_fold = total - total_not_last
        count_per_fold[sense] = (not_last_fold, last_fold)
    index_by_sense = {}
    sense_index_counter = {}
    for sense in senses:
        index_by_sense[sense] = []
        sense_index_counter[sense] = 0
    for idx,sample in enumerate(corpus):
        index_by_sense[sample[1]].append(idx)
    indexes = []
    last_fold_counter = 0
    for fold in range(k):
        fold_indexes = []
        for sense in senses:
    #we have n lists of indexes for n senses
    #we need to subset each of the lists into appropriate fold sizes
            if last_fold_counter < (k - 1):
                current_idx = 0
                while(current_idx < count_per_fold[sense][0]):
                    fold_indexes.append(index_by_sense[sense][sense_index_counter[sense]])
                    current_idx += 1
                    sense_index_counter[sense] += 1
            else:
                current_idx = 0
                while(current_idx < count_per_fold[sense][1]):
                    fold_indexes.append(index_by_sense[sense][sense_index_counter[sense]])
                    current_idx += 1
                    sense_index_counter[sense] += 1
        last_fold_counter += 1
        indexes.append(fold_indexes)
    return indexes

#Takes the corpus and indicies of the folds. Returns the corpus split by folds
def split_data_into_folds(corpus, indicies):
    folds_list = []
    corpus_series = _pd.Series(corpus)
    #convert list to series for easy subsetting
    for i,idx_list in enumerate(indicies):
        #for every fold of indexes
        folds_list.append(corpus_series[idx_list])
        #append the corresponding samples to folds_list
    return _pd.Series(folds_list)

#takes corpus and senses, returns a dictionary of dictionaries(counters)
#outside dict key is sense (meaning there is a dictionary for each sense) whose value is a dictionary(counter)
#nested dict key is word and value is the count of that word
def create_bag_of_words(corpus, senses):
    unique_word_counts = _Counter()
    senses_counter = _Counter()
    bow_dict = {}
    for sense in senses:
        bow_dict[sense] = _Counter()
        #make a Counter obj for every word sense
    for instance in corpus:
        #for every sample
        for word in instance[2]:
            #for every word in sample
            bow_dict[instance[1]][word] += 1
            #incrament the count of this word for this sense
            unique_word_counts[word] += 1
            #incrament the times we've seen this word total
        senses_counter[instance[1]] += 1
        #incrament the number of times we've seen this sense
    return bow_dict, senses_counter, unique_word_counts

#bag_dict -> dictionary{key:sense  value:Counter{key:word value:count_of_word}} -> returns count of word for sense
#test_set[n] -> Series[n][0]:sample_num  series[n][1]:actual_sense  series[n][2]:list of words
#senses -> list of senses
#num_training -> num training samples
#examples_per_sense = dictionary{key:sense  value:count_for_sense}
    #sum(examples_per_sense) = len(training_set)
#unique_words_count = dictionary{key:word  value:count_independant_of_sense}
    #len(unique_words_count) = number of unique words in training set
def new_naive_bayes(bag_dict, test_set, senses, num_training, examples_per_sense, unique_words_count, fold_num):
    #p(sense | words)  = p(sense) * p(word[0] | sense) * ... * p(word[n] | sense)
    
    #p(sense)          = count(examples_per_sense) / num_training_examples
    #p(word | sense)   = (count(word, sense) + 1) / (count(sense) + V)
    
    #so we need the following counts:
    #count(examples_per_sense)  ->  examples_per_sense[sense]
    #num_training_examples      ->  num_training
    #count(word, sense)         ->  bag_dict[sense][word]
    #count(sense)               ->  word_count_by_sense[sense]
    #V                          ->  _v
    
    #we must construct count(sense)
    word_count_by_sense = {}
    for sense in senses:
        word_count_by_sense[sense] = sum(bag_dict[sense].values())
        #the total number of words per sense is the sum of the counts of each word
    #end of construct count(sense)
    #we must construct V
    _v = len(unique_words_count)
    #end of construct V
    file_output_set = []
    file_output_set.append('fold ' + fold_num)
    answer_set = []
    #answer_set[0]=sample_num  answer_set[1]=actual_output  answer_set[2]=predicted_output
    for sentence in test_set:
        prob_sense = {}
        for sense in senses:
            prob_sense[sense] = _log(examples_per_sense[sense] / num_training_examples, 10)
            for word in sentence[2]:
                prob_sense[sense] += _log((bag_dict[sense][word] + 1) / (word_count_by_sense[sense] + _v), 10)
        prediction = max(prob_sense, key=prob_sense.get)
        answer_set.append((sentence[0], sentence[1], prediction))
        file_output_set.append((sentence[0],prediction))
    print('fold',fold_num, str(round(calculate_accuracy(answer_set)*100, 2)),'%')
    return answer_set, file_output_set

#calculates the average for a fold or the average accross all folds depending on what is passed to it
def calculate_accuracy(result):
    num_samples = len(result)
    num_correct = 0
    wrong = []
    for sample in result:
        if sample[1] == sample[2]:
            num_correct += 1
        else:
            wrong.append(sample[0])
    return num_correct/num_samples

#this writes the output to file
def write_to_file(_disamb_word, _output_tup):
    out_file = _disamb_word + '.out'
    template = "{0:15}{1:5}"
    with open(out_file, 'w') as f:
        for item in _output_tup:
            if type(item) == str:
                f.write(item+'\n')
            else:
                line_tup = (item[0], item[1])
                f.write(template.format(*line_tup)+'\n')

file_name = sys.argv
#gets filename from command line
file_name = file_name[1]
num_folds = 5
#number of folds
corpus = extract_tuples(file_name)
#extracts the sample number, the correct sense, and all the text
corpus, senses = extract_context(corpus)
#tokenizes the text for each sample and removes the sense word

indexes_of_folds = generate_k_fold_validation_indexes(corpus, senses, num_folds)
#indexes_of_folds = generate_rands(corpus, senses, num_folds)
#indexes_of_folds = even_folds(corpus, senses, num_folds)
#all three of these work, generate_k_fold_valid generatese the folds in accordance with how they were presented in class

corpus_series_of_folds = split_data_into_folds(corpus, indexes_of_folds)
#this function takes the indexes and splits the data into it's folds

results = []
write_to_file_list = []
#just some containers to hold answers

for idx, fold in enumerate(corpus_series_of_folds):
    #for every fold in the set
    train_set = _pd.Series()
    test_set = fold
    #the testing set for this iteration is 'fold'
    for i,fld in enumerate(corpus_series_of_folds):
        #Go through the folds
        if i != idx:
            train_set = _pd.concat([train_set,fld], ignore_index=True)
            #the training set is everything that is not the test set
    num_training_examples = len(train_set)
    word_freq_dict, examples_per_sense, unique_words_count = create_bag_of_words(train_set,senses)
    #creates a dictionary whose key is a sense and whose value is another dictionary (counter->dict subclass)
    #the nested dictionary has key word and value count
    this_fold_answers, file_output = new_naive_bayes(word_freq_dict, test_set,senses, num_training_examples, examples_per_sense, unique_words_count, str(idx))
    results.extend(this_fold_answers)
    write_to_file_list.extend(file_output)
accuracy = calculate_accuracy(results)
write_to_file(file_name, write_to_file_list)
print(file_name, str(round(accuracy*100, 2)),'%')
