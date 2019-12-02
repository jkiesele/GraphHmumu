

from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np
import uproot



    
    
class TrainData_graphmu(TrainData):
    def __init__(self):
        TrainData.__init__(self)
       
       
    def zeroPadOrCut2D(self, jaggedarray, maxelements, inner_shape=[]):
        '''
        Input MUST be B x H x F , F>=1
        '''
        a = jaggedarray#.array()
        a = np.array(a)
        
        nbatch = len(a)
        
        if not len(inner_shape):
            inner_shape = [len(a[0][0])]
        print(inner_shape)
        
        newarr = np.zeros([nbatch]+[maxelements]+inner_shape, dtype='float32')
        # @jit(nopython=True)
        def do_loop(nbatch, newarr, a, maxelements):
            for bi in range(nbatch):
                thislen = len(a[bi])
                #print(bi, thislen, a[bi])
                
                for e in range(maxelements):
                    if e < thislen:
                        #print(a[bi][e])
                        if len(a[bi][e]):
                            newarr[bi][e]=a[bi][e]
                    else:
                        break
            return newarr
        return do_loop(nbatch, newarr, a, maxelements)
    
    def zeroPadOrCut1D(self, jaggedarray, maxelements, makemask=False):
        '''
        Input MUST be B x H x F , F>=1
        '''
        a = jaggedarray#.array()
        a = np.array(a)
        
        nbatch = len(a)
        
        inner_shape = [1]
        print(inner_shape)
        newarr = np.zeros([nbatch]+[maxelements]+inner_shape, dtype='float32')
        
        
        # @jit(nopython=True)
        def do_loop(nbatch, newarr, a, maxelements):
            for bi in range(nbatch):
                thislen = len(a[bi])
                #print(bi, thislen, a[bi])
                
                for e in range(maxelements):
                    if e < thislen:
                        #print(a[bi][e])
                        newarr[bi][e]=a[bi][e]
                    else:
                        break
            return newarr
        return do_loop(nbatch, newarr, a, maxelements)


    def makeArray(self,a):
        newarr = np.zeros((a.shape[0],a[0].shape[0]), dtype='float32')
        for bi in range(a.shape[0]):
            newarr[bi]=a[bi]
        return newarr
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        
        tree = uproot.open(filename)["mytree/tree"]
        nevents = tree.numentries
        
        genmup = self.makeArray(np.array(tree["genmup"].array()))
        
        muonp = self.makeArray(np.array(tree["muonp"].array()))
        
        othermuonproperties = self.makeArray(np.array(tree["muonprop"].array()))
        
        print('muonp',muonp.shape)
        print('genmup',genmup.shape)
        print('othermuonproperties',othermuonproperties.shape)
        
        hits     = self.zeroPadOrCut2D(tree["hits"].array(), 100)          
        hittype  = self.zeroPadOrCut1D(tree["hittype"].array(), 100)
        hitmatched  = self.zeroPadOrCut1D(tree["hitmatched"].array(), 100)
        hiterrxx = self.zeroPadOrCut1D(tree["hiterrxx"].array(), 100)
        hiterrxy = self.zeroPadOrCut1D(tree["hiterrxy"].array(), 100)
        hiterryy = self.zeroPadOrCut1D(tree["hiterryy"].array(), 100)
        trackpos = self.zeroPadOrCut2D(tree["trackpos"].array(), 100)
        
        
        in_tiledmup = np.expand_dims(np.concatenate([muonp, othermuonproperties],axis=-1), axis=1)
        print(in_tiledmup.shape)
        in_tiledmup = np.tile(in_tiledmup, [1,hitmatched.shape[1],1]) # B x V x 1
        print(in_tiledmup.shape)
        tiledmup = in_tiledmup * hitmatched
        
        print(tiledmup.shape)
        
        
        segmudr = self.zeroPadOrCut1D(tree["segmudr"].array(), 50,)
        segmudrerr = self.zeroPadOrCut1D(tree["segmudrerr"].array(), 50)
        segx = self.zeroPadOrCut1D(tree["segx"].array(), 50)
        segy = self.zeroPadOrCut1D(tree["segy"].array(), 50)
        segxerr = self.zeroPadOrCut1D(tree["segxerr"].array(), 50)
        segyerr = self.zeroPadOrCut1D(tree["segyerr"].array(), 50)
        trackmuposx = self.zeroPadOrCut1D(tree["trackmuposx"].array(), 50)
        trackmuposy = self.zeroPadOrCut1D(tree["trackmuposy"].array(), 50)
        trackmuposxerr = self.zeroPadOrCut1D(tree["trackmuposxerr"].array(), 50)
        trackmuposyerr = self.zeroPadOrCut1D(tree["trackmuposyerr"].array(), 50)
        trackmupostation = self.zeroPadOrCut1D(tree["trackmupostation"].array(), 50)
        
        #make muon and tracker hits the same feature size by zero padding
   
        hitfeat = np.concatenate([
            hits,
            hittype,
            hiterrxx,
            hiterrxy,
            hiterryy,
            trackpos,
            tiledmup,
            hitmatched, #[-1]
            ],axis=-1)
        
        print(hitfeat.shape)
        
        muhitfeat = np.concatenate([
            segmudr,
            segmudrerr,
            segx,
            segy,
            segxerr,
            segyerr,
            trackmuposx,
            trackmuposy,
            trackmuposxerr,
            trackmuposyerr,
            trackmupostation
            ],axis=-1)
        
        print(muhitfeat.shape)
        
        pad = np.zeros([muhitfeat.shape[0],muhitfeat.shape[1], hitfeat.shape[2]-muhitfeat.shape[2] ] ,dtype='float32')
        
        muhitfeat = np.concatenate([muhitfeat, pad],axis=-1)
        
        feat = np.concatenate([hitfeat, muhitfeat],axis=1)
        
        mask = np.expand_dims(np.sum(feat, axis=-1), axis=2) # B x V x 1
        mask = np.where(mask == 0, mask, np.zeros_like(mask)+1.)
        
        truth = np.concatenate([genmup,muonp], axis=-1)
        
        hitmatched = np.concatenate([hitmatched, np.zeros((hitmatched.shape[0],segmudr.shape[1],1), dtype='float32')],axis=1 )
        
        print('hitmatched', hitmatched.shape)
        print('feat.shape',feat.shape)
        
        #clean up for unphysical values
        feat[~np.isfinite(feat)] = 0.
        feat[np.isnan(feat)] = 0.
        mask[~np.isfinite(mask)] = 0.
        mask[np.isnan(mask)] = 0.
        hitmatched[~np.isfinite(hitmatched)] = 0.
        hitmatched[np.isnan(hitmatched)] = 0.
        
        truth[~np.isfinite(truth)] = 0.
        truth[np.isnan(truth)] = 0.
        
        return [feat,mask,hitmatched] , [truth],[]
        
        
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        pass
        
