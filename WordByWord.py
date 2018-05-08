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
sequences that aren't in the to-be-provided corpus. Also, putting in info about
the expected direction of dependents would reduce the number of dim. Finally,
after calculating harmonies, could eliminate very low-harmony centers.

For now at least, don't use root/apex node

Init state w/ all EMPTYs should have harmony of ~0.25 so stable, but not too...

How to handle optional dependents?

Later, add visualization by plotting overlap between current state and centers
"""

import yaml
from itertools import product
import numpy as np
# from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns
from dynamics import calc_harmony, iterate, euclid_stop, vel_stop
from scipy.optimize import minimize  # fmin, basinhopping


class Struct(object):
    def __init__(self, lex_file=None, features=None, max_sent_length=10,
                 missing_link_cost=0.01, gamma=0.25,
                 stopping_crit='euclid_stop'):
        self.max_sent_length = max_sent_length
        self.ndim_per_position = 0
        # Maximum number of possible dependents; change to be fn. that calc.s
        # after reading in lex.
        self.ndep = 2
        self.max_links = self.max_sent_length - 1
        # Multiplier for missing links
        self.missing_link_cost = missing_link_cost
        self.gamma = gamma
        if stopping_crit == 'vel_stop':
            self.stopping_crit = vel_stop
        else:
            self.stopping_crit = euclid_stop

        self.tau = 0.01  # Time step for discretized dynamics
        self.max_time = 2000  # Max. number of time steps
        self.noise_mag = 0.001  # default
        self.tol = 0.05  # Stopping tolerance on each dim.

        if features is None:
            self.features = ['Det', 'N', 'V', 'sg', 'pl']
            self.nfeatures = len(self.features)
        else:
            self.features = features
            self.nfeatures = len(features)

        if lex_file is not None:
            self.lexicon = self._import_lexicon(lex_file)
            pf = []
            for w in self.lexicon:
                pf.append(self.lexicon[w]['phon_form'])
            self.phon_forms = list(dict.fromkeys(pf))
            self.nwords = len(self.lexicon)
            self.nphon_forms = len(self.phon_forms)
            self.pos_names = self._name_pos_dims()
            self.link_names = self._name_links()
            self.dim_names = self.pos_names + self.link_names
            self.ndim = len(self.dim_names)
            self.idx_words = {j: i for i, j in enumerate(self.lexicon.keys())}
            self.idx_phon_feat = slice(0, self.nphon_forms)
            self.idx_phon_dict = {j: i for i, j in enumerate(self.phon_forms)}
            self.idx_head_feat = slice(self.nphon_forms, self.nphon_forms
                                       + self.nfeatures)
            self.idx_links = slice(len(self.pos_names), len(self.dim_names))
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
            word_phon = self.lexicon[word]['phon_form']
            phon = [0.] * self.nphon_forms
            phon[self.idx_phon_dict[word_phon]] = 1.0
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
        seq_names = self._name_seqs()
        seq_vecs = []
        for seq in seq_names:
            curr_seq = []
            for word in seq:
                curr_word = self.idx_words[word]
                curr_seq.extend(word_vec[curr_word])
            seq_vecs.append(curr_seq)
        self.seq_vecs = seq_vecs
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
            # Finally, remove links that aren't possible with the vocabulary
        return [link_vecs[k] for k in range(len(link_vecs)) if k not in to_rm]

    def _name_links(self):
        print('Naming links...')
        links = []
#        non_empty = {k: self.lexicon[k] for k in self.lexicon
#                     if k not in 'EMPTY'}
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
        print('Naming position dimensions...')
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

    def gen_centers(self):
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
                    # Finally, removing links to/from EMPTYs
                    if word == 'EMPTY':
                        idx = [i for i, ln in enumerate(link_names)
                               if 'W' + str(word_nr) in ln]
                        for lconfig in link_vecs:
                            if any([lconfig[j] for j in idx]):
                                to_rm.append(lconfig)
                # Excluding to_rm
                configs_to_use = [c for c in link_vecs if c not in to_rm]
                for config in configs_to_use:
                    centers.append(curr_seq + config)
        # Getting rid of duplicates
        ctuple = map(tuple, centers)
        centers_unique = list(dict.fromkeys(ctuple))
        centers_array = np.array(centers_unique)
        centers_array[centers_array < 0] = 0.0  # Getting rid of -1s
        self.centers = centers_array
        print('Number of centers generated: {}'.format(centers_array.shape[0]))
        return

    def which_nonzero(self, center):
        """Returns the names of the dimensions in a cetner that are non-zero.
        """
        idx = list(np.where(center != 0)[0])
        return [self.dim_names[i] for i in idx]

    def hamming_dist(self, vec0, vec1):
        return sum(f0 != f1 for f0, f1 in zip(vec0, vec1))

    def feat_match(self, vec0, vec1):
        assert len(vec0) == len(vec1), 'Feature vectors not of equal length'
        return 1 - (self.hamming_dist(vec0, vec1) / len(vec0))

    def calculate_local_harmonies(self):
        """Cycle through the centers and use self.lexicon to look up features.
        """
        # Pos. approach: look at which links are active and where they connect
        # to. From there, look up features in self.lexicon and calculate
        # harmony in np.array
        local_harmonies = np.ones(self.centers.shape[0])
        for c, center in enumerate(self.centers):
            nonzero = self.which_nonzero(self.centers[c])
            active_links = [nonzero[i] for i, dim in enumerate(nonzero)
                            if 'L_' in dim]
            if not active_links:  # if list is empty
                local_harmonies[c] = self.missing_link_cost**self.max_links
            else:
                for link in active_links:
                    # get feat vecs
                    _, dep_word_nr, head_word_nr, head_dep = link.split('_')
                    # Just the position number
                    dep_nr = int(dep_word_nr[1])
                    dep_slice = slice(dep_nr * self.ndim_per_position
                                      + self.nphon_forms,
                                      dep_nr * self.ndim_per_position
                                      + self.nphon_forms + self.nfeatures)
                    v0 = center[dep_slice]
                    head_str = '_'.join([head_word_nr, head_dep])
                    tmp = [i for i, x in enumerate(self.pos_names) if head_str
                           in x]
                    head_slice = slice(tmp[0], tmp[0] + self.nfeatures)
                    v1 = center[head_slice]
                    local_harmonies[c] *= self.feat_match(v0, v1)
                if len(active_links) < self.max_links:
                    # Penalizing for missing links
                    local_harmonies[c] *= (self.missing_link_cost**
                                           (self.max_links -
                                            len(active_links)))
        self.local_harmonies = local_harmonies
        return

    def input_word(self, state_vec, word, pos):
        """Inputs a new word at a particular position by overwriting the values
        of the state vector at the relevant positions.
        """
        assert (pos + 1) <= self.max_sent_length, \
            'Can only add words up to max_sent_length'
        # First, get the feature vector(s) from the lexicon
        ambig_words = [w for w in self.lexicon if word in w]
        # Then, average them in case the word is ambiguous
        if len(ambig_words) != 1:
            word_vec = np.array(self.lexicon[ambig_words[0]]['head'])
        else:
            word_vec = np.zeros(self.nfeatures)
            for w in ambig_words:
                word_vec += np.array(self.lexicon[w]['head'])
            word_vec /= len(ambig_words)
        # Finally, turn on the averaged features at the correct possition
        phon = np.zeros(self.nphon_forms)
        phon[self.idx_phon_dict[word]] = 1.0
        whole_vec = np.concatenate([phon, word_vec])
        updated_state = state_vec.copy()
        start = pos*self.ndim_per_position
        stop = start + self.nphon_forms + self.nfeatures
        idx = slice(start, stop)
        updated_state[idx] = whole_vec
        return updated_state

    def neg_harmony(self, x, centers, local_harmonies, gamma):
        return -1 * calc_harmony(x, centers, local_harmonies, gamma)
    
    def jac_neg_harmony(self, x, centers, local_harmonies, gamma):
        return -1 * iterate(x, centers, local_harmonies, gamma)

    def locate_attrs(self):
        """Finds actual locations of attractors in the full harmony landscape
        using the Newton-CG algorithm on the negative of the harmony fn.
        """
        attrs = np.zeros(self.centers.shape)
        for c in range(self.centers.shape[0]):
#            print('Finding attractor for center #{}'.format(c))
#            extremum = fmin(self.neg_harmony, self.centers[c],
#                            args=(self.centers, self.local_harmonies,
#                                  self.gamma))
#            extremum = basinhopping(self.neg_harmony, self.centers[c],
#                                    T=self.gamma, stepsize=0.1,
#                                    minimizer_kwargs={'args':(self.centers,
#                                                              self.local_harmonies,
#                                                              self.gamma)})
            extremum = minimize(self.neg_harmony, self.centers[c],
                                args=(self.centers, self.local_harmonies,
                                      self.gamma), method='Newton-CG',
                                jac=self.jac_neg_harmony)
            attrs[c] = extremum.x
        unique_attrs = np.unique(np.round(attrs, 6), axis=0)
        self.attrs = unique_attrs
        print('Found {} unique attractors from {} centers'.format(
                self.attrs.shape[0], self.centers.shape[0]))
        return

    def _zero_state_hist(self):
        self.state_hist = np.zeros((self.max_time, self.ndim))

    def single_run(self, seq=None):
        """Run the model once until stopping criterion is met or
        time runs out.
        """
        assert seq is not None, 'Must provide a sequence of words.'
        self._zero_state_hist()
        self.state_hist[0, ] = self.centers[0]  # initialize to all EMPTYs
        self.energy = np.zeros(self.max_time)
        # Input the first word
        curr_pos = 0
        self.state_hist[0, ] = self.input_word(self.state_hist[0, ],
                       seq[curr_pos], curr_pos)
        # Pre-generate the noise for speed
        noise = (np.sqrt(2 * self.noise_mag * self.tau)
                 * np.random.normal(0, 1, self.state_hist.shape))
        t = 0
        while t < self.max_time-1 and seq:
            dont_stop = self.stopping_crit(self.state_hist[t], self.attrs,
                                           self.tol)
            do_stop = not dont_stop
            if dont_stop:
                self.state_hist[t+1,] = (self.state_hist[t,]
                        + self.tau * iterate(self.state_hist[t,], self.centers,
                                             self.local_harmonies, self.gamma)
                        + noise[t,])
                self.energy[t] = calc_harmony(self.state_hist[t,],
                           self.centers, self.local_harmonies, self.gamma)
                t += 1
            elif do_stop:
                if curr_pos >= self.max_sent_length-1:
                    trunc = self.state_hist[~np.any(self.state_hist == 0, axis=1)]
                    return trunc[-1]
                    break
                else:
                    print('Inputing new word')
                    curr_pos += 1
                    self.state_hist[t+1,] = (self.input_word(
                                             self.state_hist[t,],
                                             seq[curr_pos], curr_pos))
                    self.energy[t] = calc_harmony(self.state_hist[t,],
                                                   self.centers,
                                                   self.local_harmonies,
                                                   self.gamma)
                    t += 1
        trunc = self.state_hist[~np.any(self.state_hist == 0, axis=1)]
        return trunc[-1]


    def plot_trace(self):
        trunc = self.state_hist[~np.any(self.state_hist == 0, axis=1)]
        plt.plot(trunc)
#        plt.ylim(-0.01, 1.01)
        plt.xlabel('Time')
        plt.ylabel('Activation')
        plt.show()


if __name__ == '__main__':
    file = './test.yaml'
    sent_len = 3
    sys = Struct(lex_file=file, features=None, max_sent_length=sent_len,
                 missing_link_cost=0.0001, gamma=0.25,
                 stopping_crit='euclid_stop')
#    sys.lexicon
#    print(sys.dim_names)
#    print('Number of dimensions', sys.ndim)
    # print(*sys.dim_names, sep='\n')
#    link_vecs = sys._prune_links()
#    print('Number of link configurations: {}'.format(len(sys._prune_links())))
#    print('Number of sequences: {}'.format(len(sys._make_seq_vecs())))
    sys.gen_centers()
    sys.calculate_local_harmonies()
#    sns.distplot(sys.local_harmonies, kde=False, rug=True)
    sys.locate_attrs()
#    idx = np.where(sys.local_harmonies > 0.8)
#    tmp = sys.centers[idx]
#    for c in tmp:
#        print(sys.which_nonzero(c))
    final = sys.single_run(['the', 'dog'])
    sys.plot_trace()
    plt.plot(sys.energy); plt.show()
    print(sys.which_nonzero(np.round(final)))
