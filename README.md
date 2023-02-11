# Threat Dictionary

Code for the paper [When danger strikes: A linguistic tool for tracking America’s collective response to threats](https://www.pnas.org/doi/10.1073/pnas.2113891119)

```
Title:      generate_threat_dictionary.py
Authors:    Snehesh Shrestha (snehesh@umd.edu)
Date:       12/06/2019
``` 


### Paper
Choi, Virginia K., Snehesh Shrestha, Xinyue Pan, and Michele J. Gelfand. "[When danger strikes: A linguistic tool for tracking America’s collective response to threats](https://www.pnas.org/doi/10.1073/pnas.2113891119)." Proceedings of the National Academy of Sciences 119, no. 4 (2022): e2113891119.

    
### Description:

This code is for the generation of the threat dictionary as described in the paper. To understand the societal effects of broadcasted threat-relevant language, we developed a threat dictionary using natural language processing (NLP) techniques. Table S1 provides the pseudocode outlining the steps in the word generation and pruning process that went into developing the threat dictionary. In developing this linguistic measure of threat words, we leveraged computational models that map out words with semantic similarity in a high-dimensional space based on their co-occurrence in each corpus, also commonly referred to in the NLP community as word embeddings. In this paper, we applied several of GloVe's word embedding models (Pennington et al., 2014), pre-trained on the following corpora: (i) Wikipedia articles, (ii) Twitter posts, and (iii) Common Crawl web page. Additionally, with word embeddings, it is possible to specify your search for the number of candidate words that are most spatially proximal to target words (i.e., the top N words). The top 100 words per model were extracted per seed word, including “threat” and related terms such as “danger” and “warning.” After this generation of a more expansive list of threat-relevant words, common words such as “the,” “of,” “all,” etc. were filtered out. The remaining words were clustered with spectral clustering to identify majority inapposite word-clusters unrelated to threats (e.g., foreign words, numeric form of years, and named entities) via a process of full inter-rater agreement by the study authors. Words that converged across the filtered outputs of all three models produced the final threat dictionary. All terms found in the dictionary are listed in Table S2. Following these measurement development steps; the threat dictionary was applied toward building indices for measuring threat levels over time from two timestamped and geolocated corpora (U.S. newspapers and tweets during COVID-19). See paper for more details.

### Prerequisite:

1. `Python 3.7.9` : Tested and verified. While this code should work with any Python version, some dependencies packages break compatibility. If you get it to work with `Python 3.11.x or later`, please make a pull request.
2. `Mac` and `Ubuntu` : Tested and verified. While it should work with `Windows` as well, I tested it a long time ago so cannot guarantee that it will work.


### Usage:

1. Download the GloVe word embeddings from https://nlp.stanford.edu/projects/glove/ and unzip them into the data folder
2. Install the required packages using pip install -r requirements.txt
3. Modify the FILES variable to include the list of files of the seed words you wish to generate your dictionary for
4. Run the code in three steps as described below.
5. Step 1: Generate words clusters
6. Step 2: Manually filter the clusters as indicated in the paper
7. Step 3: Generate the final dictionary across all corpus


### Citation (Bibtex):
```
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
```
