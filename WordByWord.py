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

For now at least, don't use root/apex node and don't allow fragmentary
structures (obviates need for null attch.)

Init state w/ all EMPTYs should have harmony of 0.25 so stable, but not too...

Later, add visualization by plotting overlap between current state and centers
"""

import yaml
from itertools import product
import numpy as np
#from numba import jit


class Struct(object):
    def __init__(self, lex_file=None, features=None, max_sent_length=10):
        self.max_sent_length = max_sent_length
        self.ndim_per_position = 0
        # Maximum number of possible dependents; change to be fn. that calc.s
        # after reading in lex.
        self.max_deps = 2
        self.max_links = self.max_sent_length - 1

        if features is None:
            self.features = ['Det', 'N', 'V', 'sg', 'pl']
            self.nfeatures = len(self.features)
        else:
            self.features = features
            self.nfeatures = len(features)

        if lex_file is not None:
            self.lexicon = self._import_lexicon(lex_file)
            self.nwords = len(self.lexicon)
            self.dim_names = self._get_dim_names()
            self.ndim = len(self.dim_names)
            self.word_vecs = self._make_word_vecs()
        else:
            print('No lexicon loaded')
            self.lexicon = dict()
            self.nwords = 0
            self.dim_names = None
            self.ndim = None

    def _import_lexicon(self, file):
        with open(file, 'r') as stream:
            lex = yaml.safe_load(stream)
        assert 'EMPTY' in lex.keys(), 'Lexicon must include EMPTY.'
        return lex

    def _make_word_vecs(self):
        """Builds word vecs, return them in a NumPy array
        """
        word_list = []
        for word in self.lexicon:
            curr = []
            phon = [0.] * self.nwords
            phon[self.idx_phon[word]] = 1.0
            curr.extend([i for i in phon])
            curr.extend(self.lexicon[word]['head'])
            curr_ndeps = len(self.lexicon[word])
            if self.lexicon[word]['dependents'] is None:
                curr.extend([-1.] * self.max_deps * self.nfeatures)
            else:
                for dep in self.lexicon[word]['dependents']:
                    curr.extend(self.lexicon[word]['dependents'][dep])
                ndeps = len(self.lexicon[word]['dependents'])
                if ndeps > 0:
                    curr.extend([-1.] * (self.max_deps - ndeps) * self.nfeatures)
            word_list.append(curr)
        return(np.array(word_list))

    def _find_possible_sequences(self):
        """Finds all word sequences up to max_sentence_lengths. The centers
        will be these with allowed link combinations appended (done later).
        """
        # One approach: for each possible sequence of words, find all allowed
        # feature/link combinations.
        non_empty = {k: self.lexicon[k] for k in self.lexicon
                     if k not in 'EMPTY'}
        # For storing all possible sequences of words
        word_seqs = []
        # Manually adding the empty sequence
        word_seqs.append(['EMPTY'] * self.max_sent_length)
        for i in range(self.max_sent_length):
            pr = product(non_empty, repeat=i+1)
            word_seqs.extend([list(x) for x in pr])
        for i in range(len(word_seqs)):
            curr_len = len(word_seqs[i])
            if curr_len < self.max_sent_length:
                word_seqs[i].extend(['EMPTY'] * (self.max_sent_length
                                                 - curr_len))
        return word_seqs

    def _make_seq_vecs(self):
        """Returns a list of sequences and a 2D array of sequence vectors
        in which each row holds word vectors concatenated together.
        """
        word_vec = self._make_word_vecs()
#        print(word_vec)
        seqs = self._find_possible_sequences()
        seq_vecs = []
        for seq in seqs:
            curr_seq = []
            for word in seq:
                curr_word = self.idx_phon[word]
                curr_seq.extend(word_vec[curr_word,])
            seq_vecs.append(curr_seq)
        return(seqs, np.array(seq_vecs))

    def _gen_nlinks_vectors(self, link_names, nlinks):
        """Returns all of the link activation vectors that have only nlinks.
        """
        # Generates the cartesian product of [0, 1] with itself repeated as
        # many times as needed, i.e., for the length of link_names
#        too_many_links = list(map(list, product([0, 1],
#                                                repeat=len(link_names))))
        too_many_links = product([0, 1], repeat=len(link_names))
        # Gives only the link activation vectors with fewer than nlinks
#        pruned = list(filter(lambda y: sum(y) <= nlinks, too_many_links))
#        pruned = filter(lambda y: sum(y) <= nlinks, too_many_links)
        pruned = [x for x in too_many_links if sum(x) <= nlinks]
#        links = gen_nlinks_vectors(link_names, self.nlinks)
        return(pruned)

    def _prune_links(self):
        """Returns an array of link vectors after removing the ones disallowed
        under the constraints of SOSP
        """
        link_names = self._name_links()
        link_vecs = self._gen_nlinks_vectors(link_names, self.max_links)
        # A little kludgy, but works for now...
        if self.max_sent_length == 2:
            return(link_vecs)
        to_rm = []
        for i, lvec in enumerate(link_vecs):
            # Remove vectors that have the same word attached twice as a dep.
            for word_nr in range(self.max_sent_length):
                dim_per_word = self.max_deps * (self.max_sent_length-1)
                init = word_nr*dim_per_word
                idx = slice(init, init+dim_per_word)
                if sum(lvec[idx]) >= self.max_links:
                    to_rm.append(i)
                # Next, rm vectors with more than one thing attached to the
                # same dep attch site.
                for dep in ['d0', 'd1']:
                    word_str = 'W' + str(word_nr) + '_' + dep
                    dep_idx = [j for j, w in enumerate(link_names)
                               if word_str in w]
                    if sum([lvec[k] for k in dep_idx]) >= self.max_links:
                        to_rm.append(i)
            #Finally, remove links that aren't possible with the vocabulary
        return([link_vecs[k] for k in range(len(link_vecs)) if k not in to_rm])

    def _name_links(self):
        links = []
        non_empty = {k: self.lexicon[k] for k in self.lexicon
                     if k not in 'EMPTY'}
        for pos_nr in range(self.max_sent_length):
            other_positions = [x for x in range(self.max_sent_length)
                               if x != pos_nr]
            # Any word can appear at any position, so use whole lexicon here
            for op in other_positions:
                for dep in ['d0', 'd1']:  # first and second dependents
                    links.append('_'.join(['L', 'W' + str(pos_nr),
                                           'W' + str(op), dep]))
        # Calculating number of active link patterns
#        for i in range(self.max_sent_length):
#        
        return(links)

#######
    def _dim_idx(self):
        # Work out number of dimensions and way of indexing them
        npos_word_pairs = self.nwords**self.max_sent_length
        word_pos_idx = {'w'+str(i): None for i in range(self.max_sent_length)}
        for pos in word_pos_idx:
            word_pos_idx[pos] = {word: None for word in self.lexicon}
        i = 0
        for pos in word_pos_idx:
            for word in word_pos_idx[pos]:
                word_pos_idx[pos][word] = i
                i += 1
                print(i)
        print(npos_word_pairs, word_pos_idx)

    def _create_treelet_vecs(self):
        """Will return self.ndim_per_position-dimensional vectors, one for
        each word. These can then be plugged into the larger ndim vectors for
        creating the centers.

        Need to set phon form, get head features, check for dependents, if
        they exist, then get their features, all while making sure the
        position vectors have the right number of dimensions.
        """
        word_vecs = []
        for word in self.lexicon:
            word_vec = [0.] * len(self.lexicon)
            word_vec[self.idx_phon[word]] = 1.
            word_vec.extend(self.lexicon[word]['head'])
            if self.lexicon[word]['dependents'] is not None:
                for dep in self.lexicon[word]['dependents']:
                    word_vec.extend(self.lexicon[word]['dependents'][dep])
            if len(word_vec) != self.ndim_per_position:
                word_vec.extend([0.] * (self.ndim_per_position
                                        - len(word_vec)))
            word_vecs.append(word_vec)
        #return np.array(word_vecs)
#        return xr.DataArray(word_vecs, coords=)

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
        self.ndim_per_position = len(per_position)
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

        self.idx_phon = {j: i for i, j in enumerate(self.lexicon.keys())}
        self.idx_head_feat_start = len(self.lexicon)
        self.idx_head_feat_end = self.idx_head_feat_start + len(self.features)

        return all_names

    def _find_allowed_centers(self):
        """Will return a NumPy array with a center on each row.

        Note: need to create 2 different centers when there's a 0.5 in the vec
        Also: how to handle partial parses and short sentences?
        
        FUCK. THIS IS ALSO A MESS NOW.
        """
        seqs = self._find_possible_sequences()
        link_labels = self.dim_names[self.nfeat_dims:]
        clist = []  # For holding the centers
        for seq in seqs:
            seq_vec = []
            for word in seq:
                seq_vec.extend(self.word_vecs[self.idx_phon[word], :])
            if len(seq_vec) != self.ndim:
                seq_vec.extend([0.] * (self.ndim - len(seq_vec)))
            # Adds low-harmony, no-link structs to list of centers
            clist.append(seq_vec)
            # Now use seq to constrain which links can form
            if seq[0] is not 'EMPTY':
                for word_nr, word in enumerate(seq):
                    sw = '_'.join(['L', 'W' + str(word_nr), word])
                    rel_links = [x for x in link_labels if x.startswith(sw)]
                    for link in rel_links:
                        curr_vec = seq_vec.copy()
                        curr_idx = self.dim_names.index(sw + '_')
                        curr_vec[curr_idx] = 1.
                        clist.append(curr_vec)
        return clist

    def _calculate_local_harmonies(self):
        """Cycle through the links and use self.lexicon to look up features.
        """

    def _find_actual_attr_locations(self):
        """Use Newton's method (?) to find actual locations of attractors
        in the full harmony landscape.
        """
        return


if __name__ == '__main__':
    file = './test.yaml'
    sys = Struct(lex_file=file, features=None, max_sent_length=4)
#    sys.lexicon
#    print(sys.dim_names)
#    print('Number of dimensions', sys.ndim)
    # print(*sys.dim_names, sep='\n')
    link_vecs = sys._prune_links()
