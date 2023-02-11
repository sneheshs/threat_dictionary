# 
 # This file is part of the GPLv3 distribution (https://github.com/sneheshs/threat_dictionary).
 # Copyright (c) 2022 Snehesh Shrestha.
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #

'''
    Paper:
    Choi, Virginia K., Snehesh Shrestha, Xinyue Pan, and Michele J. Gelfand. "When danger strikes: 
    A linguistic tool for tracking America's collective response to threats." Proceedings of the 
    National Academy of Sciences 119, no. 4 (2022): e2113891119.
    
    Title:      generate_threat_dictionary.py
    Authors:    Snehesh Shrestha (snehesh@umd.edu)
    Date:       12/06/2019
    
    Description:
        This code is for the generation of the threat dictionary as described in the paper. 
        To understand the societal effects of broadcasted threat-relevant language, we 
        developed a threat dictionary using natural language processing (NLP) techniques. 
        Table S1 provides the pseudocode outlining the steps in the word generation and 
        pruning process that went into developing the threat dictionary. In developing this 
        linguistic measure of threat words, we leveraged computational models that map out 
        words with semantic similarity in a high-dimensional space based on their co-occurrence 
        in each corpus, also commonly referred to in the NLP community as word embeddings. 
        In this paper, we applied several of GloVe's word embedding models (Pennington et al., 2014), 
        pre-trained on the following corpora: (i) Wikipedia articles, (ii) Twitter posts, and 
        (iii) Common Crawl web page. Additionally, with word embeddings, it is possible to specify 
        your search for the number of candidate words that are most spatially proximal to target 
        words (i.e., the top N words). The top 100 words per model were extracted per seed word, 
        including “threat” and related terms such as “danger” and “warning.” After this generation 
        of a more expansive list of threat-relevant words, common words such as “the,” “of,” “all,” 
        etc. were filtered out. The remaining words were clustered with spectral clustering to 
        identify majority inapposite word-clusters unrelated to threats (e.g., foreign words, numeric 
        form of years, and named entities) via a process of full inter-rater agreement by the study 
        authors. Words that converged across the filtered outputs of all three models produced the 
        final threat dictionary. All terms found in the dictionary are listed in Table S2. Following 
        these measurement development steps; the threat dictionary was applied toward building 
        indices for measuring threat levels over time from two timestamped and geolocated corpora 
        (U.S. newspapers and tweets during COVID-19). See paper for more details.
        
    Usage:
        1. Download the GloVe word embeddings from https://nlp.stanford.edu/projects/glove/ and unzip them into the data folder
        2. Install the required packages using pip install -r requirements.txt
        3. Modify the FILES variable to include the list of files of the seed words you wish to generate the your dictionary for
        4. Run the code in three steps as described below.
        5. Step 1: Generate words clusters
        6. Step 2: Manually filter the clusters as indicated in the paper
        7. Step 3: Generate the final dictionary across all corpus
        
    Citation (Bibtex):
        @article{choi2022danger,
            title={When danger strikes: A linguistic tool for tracking America's collective response to threats},
            author={Choi, Virginia K and Shrestha, Snehesh and Pan, Xinyue and Gelfand, Michele J},
            journal={Proceedings of the National Academy of Sciences},
            volume={119},
            number={4},
            pages={e2113891119},
            year={2022},
            publisher={National Acad Sciences}
        }
'''

import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")

###################################################
###                   CONFIG                    ###
###################################################
CORPORA                 =   {'GloveWikipedia': 200, 'GloveTwitter': 200, 'GloveInternet': 300}

'''
CORPORA DIMENTIONS
Glove Twitter     25 50 100 200
Glove Wikipedia   50 100 200 300
Glove Internet    300
'''
# Defaults for experiments (using the smallest corpus)
CORPUS_DIMENSION        =   25
CORPUS 		            =   'GloveTwitter' #'GloveInternet' 'GloveTwitter' 'GloveWikipedia'

TOP_N_WORDS_PER_SEED    =   100     # Top n words you wish to generate per seed word
CLUSTER_ASSIGN_FACTOR   =   0.1     # Take 10% of the list of words to be the number of clusters
TOP_N_PER_CLUSTER       =   10      # Final top N words per cluster
PLOT_SHOW_ON_SCREEN     =   True    # For showing plot on screen
PLOT_SAVE_TO_FILE       =   False   # For saving plot into file
EXT                     =   '.txt'
ROOT_PATH               =   'data/'
INPUT_PATH              =   ROOT_PATH + 'seedwords/'
LOAD_MODEL              =   True     # Set to false to debug (loading the model takes a long time)

if CORPUS == 'GloveWikipedia':
    path_output_type = 'glove.wiki/'
elif CORPUS == 'GloveTwitter':
    path_output_type = 'glove.twitter/'
elif CORPUS == 'GloveInternet':
    path_output_type = 'glove.internet/'
else:
    path_output_type = ''


###################################################
###         MODIFY THIS SECTION                 ###
###################################################
'''
List of files containing the seed words separated by comma ['file1', 'file2', ...]
List your seed words in the files separated by new line
Only use single word seed words (some hyphenated words are allowed as long as they exists in the corpus vocabulary)
'''
FILES                   =   ['Seed_General_Threat']
path_output             =   'results/'
#--------------------------------------------------
## Step 1: Generate threat words clusters
STEP                    =   1
#--------------------------------------------------
## Please manually filter the clusters as indicated in the paper
STEP                    =   2
#--------------------------------------------------
## After manually filtering the clusters, generate the final threat dictionary across all corpus
# STEP                    =   3


###################################################
###                   FUNCTIONS                 ###
###################################################
def load_language_model(CORPUS):
    if CORPUS=='GloveWikipedia':
        print('Loading GloVe Wikipedia Corpus...')
        # w2v Stanford GloVe model
        filename = ROOT_PATH + 'glove.6B/glove.6B.' + str(CORPUS_DIMENSION) + 'd.txt'
        if not os.path.isfile(filename + '.word2vec'):
            print('GloVe Word2Vec file not found, generating...')
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove2word2vec(filename, filename + '.word2vec')
            print('Completed!')
        model = KeyedVectors.load_word2vec_format(filename + '.word2vec', binary=False)
        print('Loading Done')

    elif CORPUS=='GloveTwitter': #glove.twitter.27B
        print('Loading GloVe Twitter Corpus...')
        filename = ROOT_PATH + 'glove.twitter.27B/glove.twitter.27B.' + str(CORPUS_DIMENSION) + 'd.txt'
        if not os.path.isfile(filename + '.word2vec'):
            print('GloVe Word2Vec file not found, generating...')
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove2word2vec(filename, filename + '.word2vec')
            print('Completed!')
        model = KeyedVectors.load_word2vec_format(filename + '.word2vec', binary=False)
        print('Loading Done')

    elif CORPUS=='GloveInternet': #glove.42B
        print('Loading GloVe Common Crawl Internet Corpus...')
        # w2v Stanford GloVe model
        filename = ROOT_PATH + 'glove.42B/glove.42B.' + str(CORPUS_DIMENSION) + 'd.txt'
        if not os.path.isfile(filename + '.word2vec'):
            print('GloVe Word2Vec file not found, generating...')
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove2word2vec(filename, filename + '.word2vec')
            print('Completed!')
        model = KeyedVectors.load_word2vec_format(filename + '.word2vec', binary=False)
        print('Loading Done')

    else:
        print('Corpus Error: Unknown Corpus')
        model = None

    return model

def load_commonwords():
    f = open(INPUT_PATH + 'stopwords_long_list.txt', 'r')
    lines = f.read().lower().split('\n')
    f.close()
    return lines

def compute_center(vec):
    return np.mean(vec, axis=0)

def rank_centermost_words(v):
    scores = cosine_similarity([compute_center(v)], v)
    sorted_list = np.argsort(scores[0])[::-1]  # [::-1] reverses the order basically go backwards 1 item at a time
    return sorted_list

def format_cluster_output_filename(filename=''):
    txtCorpus = str(CORPUS)
    txtDim = '-d' + str(CORPUS_DIMENSION)
    txtTopN = '-t' + str(TOP_N_WORDS_PER_SEED)
    txtCluster = '' # '-c' + str(TOP_N_PER_CLUSTER)
    txtFilename = '-' + filename
    
    if filename == '':
        return path_output + txtCorpus + txtDim + txtTopN + txtCluster
    else:
        return path_output + txtCorpus + txtDim + txtTopN + txtCluster + txtFilename


###################################################
###                 PREP STEPS                  ###
###################################################
if not nltk.data.find('corpora/stopwords'):
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')

commonwords = set(stopwords.words('english')) ## common only has 179 words
custom_commonwords = load_commonwords()         ## custom has 667 words

#Make directory if doesn't exist
if not os.path.isdir(path_output):
    os.makedirs(path_output)
    

###################################################
###                 MAIN WORKFLOW               ###
###################################################
if STEP == 1:
    
    ###################################################
    ###                 LOAD MODEL                  ###
    ###################################################
    for CORPUS in CORPORA:
        CORPUS_DIMENSION = CORPORA[CORPUS]
        filename = format_cluster_output_filename('top_cluster_words') + '.csv'
        
        # Skip if file already generated
        if os.path.exists(filename):
            print(CORPUS  + ' already processed, skipping')
            continue
        
        if LOAD_MODEL:
            model = load_language_model(CORPUS)
            if model == None:
                print('Error: GloVe Model was not loaded, exiting...')
                exit()

        ###################################################
        ###              FIND TOP WORDS                 ###
        ###################################################
        final_words = []
        for file in FILES:
            print('****************** Processing ', file, " ***************************")

            # load all the words from the input file
            read_file = INPUT_PATH + file + EXT
            rf = open(read_file, 'r')
            text = rf.read()            # this reads the entire file
            lines = text.split('\n')    # separates the file into each line
            rf.close()

            # Filter out words that are not in the model
            valid_words = []
            for line in lines:
                line = line.strip()
                if line in model.vocab:
                    valid_words.append(line)

            #### Find all close words ############
            for each_word in valid_words:
                if each_word not in commonwords and each_word not in custom_commonwords:
                    result = model.most_similar(positive=[each_word], negative=[], topn=TOP_N_WORDS_PER_SEED)
                    
                    for item in result:
                        if item[0] not in commonwords and item[0] not in custom_commonwords: 
                            if item[0] not in final_words:    ## avoid duplicates
                                final_words.append(item[0])
            final_words = sorted(final_words)

        ###################################################
        ###              CLUSTER WORDS                  ###
        ###################################################
        NUM_OF_CLUSTERS = int(len(final_words) * CLUSTER_ASSIGN_FACTOR)
        # open(format_cluster_output_filename('top_words.txt'), 'w').write('\n'.join(final_words))
        # print('Total number of words: ', len(final_words))
            
        # Convert the final words into a vector embedding
        words_vector = model[final_words]

        # Clustering
        spectral = SpectralClustering(n_clusters=NUM_OF_CLUSTERS, eigen_solver='arpack', affinity="nearest_neighbors")
        spectral.fit(words_vector)
        cluster_assignments = spectral.labels_.astype(int)

        # GROUPING INTO EACH CLUSTER COLUMNS
        fw_array = [[x] for x in final_words]
        cl_array = [[x] for x in cluster_assignments]
        new_coord = np.concatenate((fw_array, cl_array), axis=1)
        df = pd.DataFrame(data=new_coord, columns=('words', 'cluster'))
        out = pd.concat({k: g.reset_index(drop=True) for k, g in df.groupby('cluster')['words']}, axis=1)

        sorted_out = out.copy()
        for col in out:
            lis = out[col].dropna().tolist()
            vec = model[lis]

            # Get words in the order of closest to the center
            sorted_list = rank_centermost_words(vec)

            for i in range(0, len(out[col])):
                if i >= len(sorted_list):
                    sorted_out[col][i] = ''
                else:
                    sorted_out[col][i] = out[col][sorted_list[i]]
                    
        sorted_out.to_csv(format_cluster_output_filename('top_cluster_words') + '.csv', index=False)

elif STEP == 2:
    print('Please review the generated words and their clusters and filter them as described in the paper with 3 or more reveiwers.')
    
elif STEP == 3:
    final_threat_dictionary = []
    all_words = {}
    
    # Load all the words from each corpus
    for CORPUS in CORPORA:
        CORPUS_DIMENSION = CORPORA[CORPUS]
        
        filename = format_cluster_output_filename('top_cluster_words') + '.csv'
        data = pd.read_csv(filename).fillna('')
        
        for col in data.iteritems():
            for word in col[1]:
                word = str(word).strip()
                if word != '':
                    if word in all_words:
                        all_words[word] += 1
                    else:
                        all_words[word] = 1
          
    # Select words that are in all the corpora       
    for word in all_words:
        if all_words[word] == len(CORPORA):
            final_threat_dictionary.append(word)
    final_threat_dictionary = sorted(final_threat_dictionary)
    
    ## SAVE THE FINAL DICTIONARY
    print('Total number of words: ', len(final_threat_dictionary))
    open(path_output + 'final_dictionary.txt', 'w').write('\n'.join(final_threat_dictionary))
