//
//  SegmentalKmeans.hpp
//  speech3
//
//  Created by Dajian on 15/10/9.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#ifndef SegmentalKmeans_hpp
#define SegmentalKmeans_hpp

#include <vector>
#include <stdio.h>
#include <math.h>
#include <string>
#include <fstream>
#include "HMMModelSGM.hpp"
#include "HMMTrainer.h"
#include "VectorOperation.hpp"
#include "Constants.h"

class SegmentalKmeans: public HMMModelSGM, HMMTrainer{
    
private:
    
    void estimateCovariances(std::vector<std::vector<MFCC_Feature> > & container); // to estimate covariance
    
public:
    
    // STEP 1: set parameters of the model
    void setParameters(int K, int FEATURE_LENGTH);
    
    // STEP 2: commit single Gaussian HMM
    void commit(const std::vector<std::vector<MFCC_Feature> > &);
    
};

#endif /* SegmentalKmeans_hpp */
