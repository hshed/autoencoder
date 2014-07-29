'''
Created on Nov 4, 2013
@author: Hrishikesh
'''

import cPickle as pickle
import numpy as np
def _chunks(x, n):
    """Produce `n`-character chunks from `s`."""
    for start in range(0, len(x), n):
        yield x[start:start+n]

class Fasta():
    def __init__(self, file_name):
        self._filename = file_name
        self._alphabet = ['A','C','D','E','F','G','H','I','K',
                          'L','M','N','P','Q','R','S','T','V','W','Y']
    
    def fastaReader(self):
        """Generator function to iterator over Fasta records (as string tuples).
            Parses fasta format file.
    
            For each reocrd in file, a tuple of two strings is returned.
            ('FASTA header','sequence')...

            ***Usage:
                for values in FastaReader(open("fastafile.fasta")):
                    ...
                    ...do something with the values...
                    ...
            ***
        """
        # check proper format => searches for ">"
        while True:
            line = self._filename.readline()
            if line == "":
                return  # check lines ahead by returning to loop
            if line[0] == ">":
                break

        while True:
            if line[0] != ">":
                raise ValueError("Incorrect Fasta format file. The file should start with '>'")
            title = line[1:].rstrip()  # remove any white space at the end of the file
            lines = []
            line = self._filename.readline()
            while True:
                if not line:
                    break
                if line[0] == ">":  # stop at next entry
                    break
                lines.append(line.rstrip())
                line = self._filename.readline()

            yield title, "".join(lines).replace(" ", "").replace("\r", "")

            if not line:
                return

    def parser(self):
        """
        ***Usage:
            for sequence, name, description in FastaParser(open("fastafile.fasta")):
            ...
            ...do something with the values...
            ...
        ***
        """
        for title, sequence in self.fastaReader():
            try:
                _name = self.get_name(title)
            except IndexError:
                assert not title, repr(title)
                _name = ""
            yield sequence, _name 
        
    def get_name(self, j):
        '''
        returns the name of the protein from fasta headers
        '''
        t = []
        j = j.rsplit()[1:]
        for i in j:
            if "OS" not in i:
                t.append(i)
            else:
                break
        s = " ".join(t[:-1])
        if "," in s:
            return s.rsplit(",")[0]
        else: return s
    
    def family_sequence(self,window = 4):
        '''This function reads each family fasta file 
            and stores the sequences in pickle format.
            
            >>>family_sequence(window = 4)
            >>>family_sequence() #window size is 4 by default'''
        _sequence = ''
        for _seq, _ in self.parser():
            _sequence = _sequence + _seq #whole sequence of the family
        
        _mat = np.empty([window, 20])
        _train_seq = len(_sequence)*70/100
        _train_seq_mat = np.empty([_train_seq/window + 1, 20*window]) #(T/w) x 20w
        print len(_train_seq_mat)
        _test_seq = len(_sequence) - _train_seq
        _test_seq_mat = np.empty([_test_seq/window, 20*window])
        for i in range(0, _train_seq, window):
            s = _sequence[i:i+window]
            if len(s)<4:
                break
            s = list(s)
            for amino in np.arange(0, len(self._alphabet)):
                for w in np.arange(0, window):
                    _mat[w][amino] = (np.squeeze(self._alphabet[amino]) == s[w]).astype(int)
            #print i
            _train_seq_mat[i/window]
            _train_seq_mat[i/window] = _mat.reshape([1,20*window])
        
        for i in range(0, _test_seq, window):
            s = _sequence[_train_seq + i:_train_seq + i+window]
            if len(s)<4:
                break
            s = list(s)
            for amino in np.arange(0, len(self._alphabet)):
                for w in np.arange(0, window):
                    _mat[w][amino] = (np.squeeze(self._alphabet[amino]) == s[w]).astype(int)
            #print i, window
            _test_seq_mat[i/window] = _mat.reshape([1,20*window])
                
        print _train_seq_mat.T.shape, _test_seq_mat.T.shape
        return [_train_seq_mat.T, _test_seq_mat.T]

if __name__ == '__main__':
    families = ['ATP8', 'CalpninMAPRE',
                    'Calponin', 'CaspaseABcL-2', 'CaspaseANP32', 'CaspasePeptidaseS1B',
                     'CBP4', 'CelluloseBinding2501', 'CIDE-N', 'FicolinLectin',
                      'GNATacetyltransferase','Lipasechaperone','SerineEsterase']
    for f in families:
        import time
        s = time.time()
        print 'Importing family ', f
        with open('database/' + f + '.fasta','r') as fi:
            fasta = Fasta(fi)
            pi = fasta.family_sequence(6)
            pickle.dump(pi[0], open( "processed/train/" + f + ".pkl", "wb" ) )
            pickle.dump(pi[1], open( "processed/test/" + f + ".pkl", "wb" ) )
    
        print time.time() - s