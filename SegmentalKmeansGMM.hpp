//
//  SegmentalKmeansGMM.hpp
//  speech3
//
//  Created by Dajian on 15/10/9.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#ifndef SegmentalKmeansGMM_hpp
#define SegmentalKmeansGMM_hpp

#include <vector>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include "HMMModelGMM.hpp"
#include "HMMTrainer.h"
#include "Constants.h"

class SegmentalKmeansGMM: public HMMModelGMM, HMMTrainer{
    
private:
    
    // initialize miu and cov (Kmeans)
    void initializeGMMParameters(std::vector<std::vector<MFCC_Feature> > & container);
    
public:
    
    // use EM to estimate GMM model and update parameters
    void estimateGMM(std::vector<std::vector<MFCC_Feature> > & container);
    
    // STEP 1: set parameters for HMM
    void setParameters(int K, int FEATURE_LENGTH, int KERNEL_NUMBER);
    
    // STEP 2: train HMM model
    void commit(const std::vector<std::vector<MFCC_Feature> > &);
    
};

#endif /* SegmentalKmeansGMM_hpp */
