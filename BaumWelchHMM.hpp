//
//  BaumWelchHMM.hpp
//  speech3
//
//  Created by Dajian on 15/12/23.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#ifndef BaumWelchHMM_hpp
#define BaumWelchHMM_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <math.h>
#include "HMMModelGMM.hpp"
#include "HMMTrainer.h"
#include "Constants.h"

class BaumWelchHMM: public HMMModelGMM, public HMMTrainer{
    
private:
    
    // split centroids whose covariance is highest and perform kmeans
    void split_kmeans(std::vector<std::vector<float> > & miu_one_state, std::vector<std::vector<float> > & cov_one_state, std::vector<MFCC_Feature> & container);
    
    // initialize miu and cov (split Kmeans)
    void initializeGMMParameters(std::vector<std::vector<MFCC_Feature> > & container);
    
public:
    
    // estimate GMM model using Baum-Welch
    float estimateGMM(const std::vector<std::vector<MFCC_Feature> > & samples,
                     // for soft calculate transition probability
                     std::vector<std::vector<std::vector<std::vector<float> > > > & buffer_alphas,
                     std::vector<std::vector<std::vector<std::vector<float> > > > & buffer_betas,
                     std::vector<std::vector<std::vector<std::vector<float> > > > & buffer_pxs,
                     // for soft calculate miu, cov, alpha
                     std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > & buffer_lambdas,
                     // the index of the model, for extract required data to calculate parameters
                     int index_Model = 0
                     );
    
    // STEP 1: set parameters for HMM
    void setParameters(int K, int FEATURE_LENGTH, int KERNEL_NUMBER);
    
    // STEP 2: train HMM model
    void commit(const std::vector<std::vector<MFCC_Feature> > &);
    
};

#endif /* BaumWelchHMM_hpp */
