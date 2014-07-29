'''
Created on Nov 4, 2013
@author: Hrishikesh
'''
import warnings as w
from Autoencoder import Autoencoder
import time

def warning_on_one_line(message, category, filename, lineno, fi=None, line=None):
    return ' %s:%s: %s:%s' % (filename, lineno, category.__name__, message)

class Trainer():
    def __init__(self, data, autoencoder):
        self.ae = autoencoder
        self.input = data
        self.target = data
        w.formatwarning = warning_on_one_line
        
    def train(self):
        start = time.time()
        print 'Training...'
        extracted_features = self.ae.fit_transform(self.input)
        print extracted_features.shape
        #print 'test', test.shape
        print 'Training took ', time.time() - start, 'seconds'
        print 'Accuracy on Training data? ', self.ae._predict(self.input)
        #print 'Accuracy on Testing data? ', self.ae._predict(test)
    
if __name__ == '__main__':
    import cPickle as pickle
    families = ['ATP8', 'CalpninMAPRE',
                    'Calponin', 'CaspaseABcL-2', 'CaspaseANP32', 'CaspasePeptidaseS1B',
                     'CBP4', 'CelluloseBinding2501', 'CIDE-N', 'FicolinLectin',
                      'GNATacetyltransferase','Lipasechaperone','SerineEsterase']
    ae = Autoencoder(max_iter=200, sparsity=0.01,beta=0.3, n_hidden=90, alpha=3e-3,verbose=False, random_state=3)
    for f in families:
        s = time.time()
        print 'family', f
        data = pickle.load( open( "processed/train/" + f + ".pkl", "rb" ) )
        #print data.shape
        trainer = Trainer(data,ae)
        trainer.train()
        print time.time(), 'ss'
        '''test = pickle.load( open( "processed/test/" + f + ".pkl", "rb" ) )
        trainer = Trainer(test, ae)
        trainer.train()'''
    '''data = pickle.load( open( "processed/" + "CaspaseABcL-2" + ".pkl", "rb" ) )
    print data.shape
    trainer = Trainer(data)
    trainer.train()'''