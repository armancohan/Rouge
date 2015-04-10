'''
Created on Apr 10, 2015

@author: rmn
'''
from __future__ import division
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
STOPWORDS = 'stopwords.txt'
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
lmtzr = WordNetLemmatizer()
tokenizer = RegexpTokenizer('[^\w\-\']+', gaps=True)
import re

with file(STOPWORDS) as f:
    stopwords = frozenset([l.strip().lower() for l in f])


class Rouge(object):

    def rouge_l(self, ref_summ, cand_summ):
        '''
        Calculates the summary level rouge_L score
            for the reference summary "ref_summ" and candidate
            summary "cand_summ"

        Args:
            ref_summ(str)
            cand_summ(str)

        Returns:
            float
        '''
        R = tokenize(ref_summ)

        cand_sents = sent_tokenize(cand_summ)
        for s in cand_sents:
            map = {0: for w in tokenize(s)}
        lcs_union = 
        
        C = tokenize(cand_summ)

    def rouge_l_sent(self, sent1, sent2, beta=1):
        '''
        LCS-Based sentence level precision, recall and F-1

        Args:
            sent1: Reference summary sentence
            sent2: Candidate summary sentence
            beta: param for f score

        Returns:
            float
        '''
        x = tokenize(sent1)
        y = tokenize(sent2)
        lcs = len(self._lcs(x, y))
        recall = lcs / len(x)
        prec = lcs / len(y)
        f1 = (1 + beta ** 2) * prec * recall / (recall + beta ** 2 + prec)
        return recall, prec, f1

    def _lcs(self, x, y):
        n = len(x)
        m = len(y)
        table = dict()  # a hashtable, but we'll use it as a 2D array here

        for i in range(n + 1):     # i=0,1,...,n
            for j in range(m + 1):  # j=0,1,...,m
                if i == 0 or j == 0:
                    table[i, j] = 0
                elif x[i - 1] == y[j - 1]:
                    table[i, j] = table[i - 1, j - 1] + 1
                else:
                    table[i, j] = max(table[i - 1, j], table[i, j - 1])

        # Now, table[n, m] is the length of LCS of x and y.

        # Let's go one step further and reconstruct
        # the actual sequence from DP table:

        def recon(i, j):
            if i == 0 or j == 0:
                return []
            elif x[i - 1] == y[j - 1]:
                return recon(i - 1, j - 1) + [x[i - 1]]
            # index out of bounds bug here: what if the first elements in the
            # sequences aren't equal
            elif table[i - 1, j] > table[i, j - 1]:
                return recon(i - 1, j)
            else:
                return recon(i, j - 1)

        return recon(n, m)


def tokenize(doc, stem=False, no_stopwords=False, lemmatize=False):
    """
    tokenizes a string

    Args:
        stem(bool)
        no_stopwords(bool): If true, there will be no stopwords
        lemmatize(bool): Does the lemmatization just for nouns

    Returns:
        list(str) 
    """
    all_digits = re.compile(r"^\d+$").search
    terms = []
    for w in tokenizer.tokenize(doc.lower()):
        if no_stopwords:
            if w not in stopwords and (not all_digits(w)):
                if lemmatize:
                    terms.append(lmtzr.lemmatize(w))
                elif stem:
                    terms.append(stemmer.stem(w))
                else:
                    terms.append(w)
        else:
            if lemmatize:
                terms.append(lmtzr.lemmatize(w))
            elif stem:
                terms.append(stemmer.stem(w))
            else:
                terms.append(w)
    return terms


def sent_tokenize(text, abbrev_list=['dr', 'vs',
                                     'etc', 'mr',
                                     'mrs', 'prof',
                                     'inc', 'et',
                                     'al', 'Fig', 'fig']):
    '''
    Tokenizes a string into sentences

    Args:
        text(str) -- The text being tokenized
        abbrev_list(list) -- a list of abbreviations followed by dot
            to exclude from tokinzation e.g. mr. ms. etc.

    Returns
        list of strings -- list of sentences
    '''
    punkt_param = PunktParameters()
    punkt_param.abbrev_types = set(abbrev_list)
    sent_detector = PunktSentenceTokenizer(punkt_param)
    return sent_detector.tokenize(text)


if __name__ == '__main__':
    rg = Rouge()
    print rg.rouge_l_sent("police killed the gunman", "police kill the gunman")
    print rg.rouge_l_sent("police killed the gunman", "the gunman kill police")
