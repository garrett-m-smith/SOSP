# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:26:04 2018

@author: garrettsmith

Word-by-word SOSP sentence processing tools

The lexicon is basically a dictionary where the keys are attachment sites
(head and dependent) and the values are NumPy arrays for the features.

A treelet is a vector of head identities (lexical forms), head features, and
dependent features.

CHANGE (04/20): Instead of lexical-item-specific features at each position,
I'll use one-hot word-identity dimensions at each position and a single bank
of features for each position. This should greatly reduce the dimensionality
of the system. ALSO: To allow for strings less than max_sent_length, need to
add NULL lexical items at each position
"""

# import numpy as np
import yaml
from itertools import product


class Struct(object):
    def __init__(self, lex_file=None, features=None, max_sent_length=10):
        self.max_sent_length = max_sent_length

        if features is None:
            self.features = ['Det', 'N', 'V', 'sg', 'pl']
        else:
            self.features = features

        if lex_file is not None:
            self.lexicon = self._import_lexicon(lex_file)
            self.dim_names = self._get_dim_names()
            self.ndim = len(self.dim_names)
        else:
            print('No lexicon loaded')
            self.lexicon = dict()
            self.dim_names = None
            self.ndim = None

    def _import_lexicon(self, file):
        with open(file, 'r') as stream:
            lex = yaml.safe_load(stream)
        assert 'EMPTY' in lex.keys(), 'Lexicon must include EMPTY.'
        return lex

    def _get_dim_names(self):
        assert self.lexicon is not None, 'Must initialize lexicon.'
        per_position = []
        for word in self.lexicon:
            per_position.append(word)
        for feat in self.features:
            per_position.append(feat)
        for word in self.lexicon:
            if self.lexicon[word]['dependents'] is not None:
                for dep in self.lexicon[word]['dependents']:
                    for feat in self.features:
                        per_position.append('_'.join([word, dep, feat]))
        links = []
        non_empty = {k: self.lexicon[k] for k in self.lexicon
                     if k not in 'EMPTY'}
        for pos_nr, word in product(range(self.max_sent_length), non_empty):
            other_positions = [x for x in range(self.max_sent_length)
                               if x != pos_nr]
            # Any word can appear at any position, so use whole lexicon here
            for op, ow in product(other_positions, non_empty):
                if self.lexicon[ow]['dependents'] is not None:
                    for dep in self.lexicon[ow]['dependents']:
                        links.append('_'.join(['L', 'W' + str(pos_nr),
                                               word, 'W' + str(op), ow,
                                               dep]))
        all_names = []
        for i in range(self.max_sent_length):
            tmp = ['W' + str(i) + '_' + pf for pf in per_position]
            for x in tmp:
                all_names.append(x)

        for i in links:
            all_names.append(i)
        self.nlinks = len(links)
        self.nfeat_dims = len(per_position) * self.max_sent_length
        return all_names

    def _find_possible_sequences(self):
        """Finds all word sequences upto max_sentence_lengths.
        """
        # One approach: for each possible sequence of words, find all allowed
        # feature/link combinations.
        non_empty = {k: self.lexicon[k] for k in self.lexicon
                     if k not in 'EMPTY'}
        word_seqs = []  # For storing all possible sequences of words
        for i in range(self.max_sent_length):
            pr = product(non_empty, repeat=i+1)
            word_seqs.extend([list(x) for x in pr])
        for i in range(len(word_seqs)):
            curr_len = len(word_seqs[i])
            if curr_len < self.max_sent_length:
                word_seqs[i].extend(['EMPTY'] * (self.max_sent_length
                                                 - curr_len))
        return word_seqs

    def _find_allowed_centers(self):
        """Will return a NumPy array with a center on each row.
        """
        # Approach: create the centers by gradually going through chunks of
        # the dimensions, drawing features from the lexicon. When you get to
        # the link dimensions, we can enforce the one-licensor constraint by
        # noting that the link attaching to the head of each word at each
        # position, e.g., L_W0_the_*, is mutually exclusive of all the other
        # L_W0_the_*. Might need to build in extra dimensions yet, though, for
        # links to the Null node...
        seqs = self._find_possible_sequences()
        return

    def _find_actual_attr_locations(self):
        """Use Newton's method (?) to find actual locations of attractors
        in the full harmony landscape.
        """
        return


if __name__ == '__main__':
    file = './test.yaml'
    sys = Struct(file, features=None, max_sent_length=3)
    sys.lexicon
#    print(sys.dim_names)
    print('Number of dimensions', sys.ndim)
    #print(*sys.dim_names, sep='\n')

# Creating 1-hot phonological form vectors: nwords = len(lex.keys())
