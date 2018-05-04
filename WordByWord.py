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
of the system. ALSO: To allow for strings less than max_sent_length, added
EMPTY lexical items at each position.

UPDATE (05/03): Now doing the same thing for the dependent features: for each
dependent, there will be a single set of features instead of, e.g, dog-specific
dependent features.

UPDATE (05/04): Ambiguous words are disambiguated in the lexicon file, but if
they share a phonological form, only a single lexeme is used for making the
dimension names.

POSSIBLITY FOR REDUCING NUMBER OF DIMENSIONS: Have a full lexicon (like now),
but if only want to consider particular sequences, add a method for removing
sequences that aren't in the to-be-provided corpus.

For now at least, don't use root/apex node and don't allow fragmentary
structures (obviates need for null attch.)

Init state w/ all EMPTYs should have harmony of ~0.25 so stable, but not too...

How to handle optional dependents?

Later, add visualization by plotting overlap between current state and centers
"""

import yaml
from itertools import product
import numpy as np
# from numba import jit


class Struct(object):
    def __init__(self, lex_file=None, features=None, max_sent_length=10):
        self.max_sent_length = max_sent_length
        self.ndim_per_position = 0
        # Maximum number of possible dependents; change to be fn. that calc.s
        # after reading in lex.
        self.ndep = 2
        self.max_links = self.max_sent_length - 1

        if features is None:
            self.features = ['Det', 'N', 'V', 'sg', 'pl']
            self.nfeatures = len(self.features)
        else:
            self.features = features
            self.nfeatures = len(features)

        if lex_file is not None:
            self.lexicon = self._import_lexicon(lex_file)
            self.phon_forms = []
            for w in self.lexicon:
                self.phon_forms.append(self.lexicon[w]['phon_form'])
            self.nwords = len(self.lexicon)
            self.pos_names = self._name_pos_dims()
            self.link_names = self._name_links()
            self.dim_names = self.pos_names + self.link_names
            self.ndim = len(self.pos_names) + len(self.link_names)
            self.idx_phon = {j: i for i, j in enumerate(self.lexicon.keys())}
            self.idx_head_feat = slice(self.nwords, self.nwords+self.nfeatures)
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
#            curr_ndeps = len(self.lexicon[word])
            if self.lexicon[word]['dependents'] is None:
                curr.extend([-1.] * self.ndep * self.nfeatures)
            else:
                for dep in self.lexicon[word]['dependents']:
                    curr.extend(self.lexicon[word]['dependents'][dep])
                ndeps = len(self.lexicon[word]['dependents'])
                if ndeps > 0:
                    # Code non-existent features as -1s...
                    curr.extend([-1.] * (self.ndep-ndeps) * self.nfeatures)
            word_list.append(curr)
        return np.array(word_list)

    def _name_seqs(self):
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
        """Returns a list of sequence vectors in which each element holds word
        vectors concatenated together.
        """
        word_vec = self._make_word_vecs()
#        print(word_vec)
        seqs = self._name_seqs()
        seq_vecs = []
        for seq in seqs:
            curr_seq = []
            for word in seq:
                curr_word = self.idx_phon[word]
                curr_seq.extend(word_vec[curr_word])
            seq_vecs.append(curr_seq)
        return seq_vecs

    def _gen_nlinks_vectors(self, link_names, nlinks):
        """Returns all of the link activation vectors that have only nlinks.
        Probably need to do some pruning before this step, because it generates
        ndep^len(link_names) vectors before pruning out the ones with too many
        links.
        """
        # Generates the cartesian product of [0, 1] with itself repeated as
        # many times as needed, i.e., for the length of link_names
        too_many_links = product([0, 1], repeat=len(link_names))
        # Gives only the link activation vectors with fewer than nlinks
#        pruned = list(filter(lambda y: sum(y) <= nlinks, too_many_links))
#        pruned = filter(lambda y: sum(y) <= nlinks, too_many_links)
        pruned = [x for x in too_many_links if sum(x) <= nlinks]
#        links = gen_nlinks_vectors(link_names, self.nlinks)
        return list(map(list, pruned))

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
                dim_per_word = self.ndep * (self.max_sent_length-1)
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
        return [link_vecs[k] for k in range(len(link_vecs)) if k not in to_rm]

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
        return links

    def _name_pos_dims(self):
        """Returns a list of the dimension names. There are always ndep
        dependents at a position regardless of what word is in that position.
        Also only creates one phonological form for ambiguous words, like
        'the_sg' and 'the_pl.'
        """
        assert self.lexicon is not None, 'Must initialize lexicon.'
        per_position = []
#        for word in self.lexicon:
        for word in self.phon_forms:
            per_position.append(word)
        for feat in self.features:
            per_position.append(feat)
        for dep in range(self.ndep):
            for feat in self.features:
                per_position.append('d' + str(dep) + '_' + feat)
#        for word in self.lexicon:
#            if self.lexicon[word]['dependents'] is not None:
#                for dep in self.lexicon[word]['dependents']:
#                    for feat in self.features:
#                        per_position.append('_'.join([word, dep, feat]))
        self.ndim_per_position = len(per_position)

        all_names = []
        for i in range(self.max_sent_length):
            tmp = ['W' + str(i) + '_' + pf for pf in per_position]
            for x in tmp:
                all_names.append(x)
        return all_names

    def _gen_centers(self):
        """Will return a NumPy array with a center on each row.

        Because links are only care about sentence position and attch. site,
        don't have to worry about what words are in the positions, except to
        make sure they allow dependents.

        Note: need to create 2 different centers when there's a 0.5 in the vec
        """
        # Notes: link vec of zeros is always possible, no matter how many words
        # have been input. No links turned on after reading first word.
        # As words come in, can only allow centers with them attching somehow
        # to previous words, not looking ahead.
        seq_vecs = self._make_seq_vecs()
        seq_names = self._name_seqs()
        assert len(seq_vecs) == len(seq_names), \
            'Number of sequence vectors mismatches number of sequence names.'
        link_names = self._name_links()
        link_vecs = self._prune_links()
        centers = []
        # Cycle through seqs and find allowed links
        for seq_name, seq in zip(seq_names, seq_vecs):
            curr_seq = seq.copy()
            if seq_name[0] == 'EMPTY':
                # Assumes 0th link vec is one with no links!
                centers.append(curr_seq + link_vecs[0])
            elif seq_name[1] == 'EMPTY':
                centers.append(curr_seq + link_vecs[0])
            else:
#                configs_to_use = link_vecs.copy()
                # Need to exclude attchs. to EMPTYs
                try:
                    first_empty = seq_name.index('EMPTY')
                    empties = ['W' + str(i) for i in
                               range(first_empty, self.max_sent_length)]
                    # Indexing the dimensions that have links to EMPTYs
                    empty_idx = [i for i, ln in enumerate(link_names) for e in
                           empties if e not in ln]
                except ValueError:
                    empty_idx = []
                to_rm = []
                for lconfig in link_vecs:
                    for i in empty_idx:
                        if lconfig[i] != 0:
                            to_rm.append(lconfig)
                # Now removing link configs if they link to a non-existent
                # dependent
                for word_nr, word in enumerate(seq_name):
                    if self.lexicon[word]['dependents'] is None:
                        null_attch = ['W' + str(word_nr) + '_' + 'd'
                                      + str(j) for j in range(self.ndep)]
                        null_idx = [i for i, ln in enumerate(link_names)
                                    for n in null_attch if n in ln]
                        for lconfig in link_vecs:
                            for i in null_idx:
                                if lconfig[i] != 0:
                                    to_rm.append(lconfig)
                    elif len(self.lexicon[word]['dependents']) < self.ndep:
                        null_attch = ['W' + str(word_nr) + '_' + 'd'
                                      + str(j) for j in
                                      range(1, self.ndep)]
                        null_idx = [i for i, ln in enumerate(link_names)
                                    for n in null_attch if n in ln]
                        for lconfig in link_vecs:
                            for i in null_idx:
                                if lconfig[i] != 0:
                                    to_rm.append(lconfig)
                configs_to_use = [c for c in link_vecs if c not in to_rm]
                for config in configs_to_use:
                    centers.append(curr_seq + config)
        # Making sure there are no duplicates
#        unique_centers = list(dict.fromkeys(centers))
        print('Number of centers generated: {}'.format(len(centers)))
        centers_array = np.array(centers)
        centers_array[centers_array < 0] = 0.0  # Getting rid of -1s
        return centers_array

    def _calculate_local_harmonies(self):
        """Cycle through the centers and use self.lexicon to look up features.
        """

    def _locate_attrs(self):
        """Use Newton's method (?) to find actual locations of attractors
        in the full harmony landscape.
        """
        return  # Array of actual attractor locations


if __name__ == '__main__':
    file = './test.yaml'
    sent_len = 4
    sys = Struct(lex_file=file, features=None, max_sent_length=sent_len)
#    sys.lexicon
#    print(sys.dim_names)
#    print('Number of dimensions', sys.ndim)
    # print(*sys.dim_names, sep='\n')
#    link_vecs = sys._prune_links()
#    print('Number of link configurations: {}'.format(len(sys._prune_links())))
#    print('Number of sequences: {}'.format(len(sys._make_seq_vecs())))
    print('Number of centers: {}'.format(len(sys._gen_centers())))
