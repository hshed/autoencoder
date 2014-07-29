'''
Created on Jan 22, 2014
@author: Hrishikesh
'''
import cPickle as pickle
class Sequence():
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

    
    def motifs(self):
        #_sequence = ''
        #motif_dict = {}
        for title, _seq in self.fastaReader():
            #_sequence = _sequence + _seq #whole sequence of the family
            motif_dict = {}
            #print 'length of current sequence', len(_seq)
            for i in range(0, len(_seq), 1):
                #if i%50 == 0:
                # print 'i', i
                motif = _seq[i:i+4]
                if(len(motif) !=4):
                    break
                if motif not in motif_dict:
                    motif_dict[motif] = 1
                else:
                    motif_dict[motif] +=1
                    #print motif_dict
            yield title, motif_dict

if __name__ == '__main__':
    families = ['uniprot_sprot']
    for f in families:
        import time
        s = time.time()
        #print 'Importing family ', f
        with open('database/' + f + '.fasta','r') as fi:
            fasta = Sequence(fi)
            f = open("processed/motif/org-wise/motif" + ".txt","w")
            for title, motifs in fasta.motifs():
                #pi = fasta.motifs()
                #title = title.split(" ")[0]
                #title = title.replace("|", " ")                
                #pickle.dump(motif, open( "processed/motif/org-wise/" + title + ".pkl", "wb" ) )
                
                
                f.write(title + '\n')
                for motif in sorted(motifs, key=motifs.get, reverse=True):
                    f.write(motif + ' :\t ' + str(motifs[motif]) + '\n')
            f.close()
        ##un commment the following after first run
        '''with open('processed/motif/uniprot_sprot.pkl','r') as fi:
            pi = pickle.load(fi)
            f = open('processed/motif/motif.txt','w')
            for motif in sorted(pi, key=pi.get, reverse=True):
                f.write(motif + ' :\t ' + str(pi[motif]) + '\n')
            f.close'''
    
        print 'time taken', time.time() - s