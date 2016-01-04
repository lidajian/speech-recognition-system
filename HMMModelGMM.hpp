//
//  HMMModelGMM.hpp
//  speech3
//
//  Created by Dajian on 15/10/22.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#ifndef HMMModelGMM_hpp
#define HMMModelGMM_hpp

#include "HMMModel.hpp"

class HMMModelGMM: public HMMModel{
    
private:
    
    void clear();
    
protected:
    
    int KERNEL_NUMBER; // # of GMM kernel
    
    std::vector<std::vector<MFCC_Feature> > * miu; // centers: K * KERNEL_NUMBER * FEATURE_LENGTH Matrix
    
    std::vector<std::vector<std::vector<float> > > * cov; // covariances: K * KERNEL_NUMBER * FEATURE_LENGTH Matrix
    
    std::vector<std::vector<float> > * alpha; // P(y | theta)

    
public:
    
    HMMModelGMM();
    
    ~HMMModelGMM();
    
    int getNumberOfKernel(){
        return KERNEL_NUMBER;
    }
    
    // log( P(v1 | state) )
    float probabilityMFCC(const std::vector<float> & v1, int state, int kernel = 0);
    
    // Gaussian probability distance function
    float distanceMFCC(const std::vector<float> & v1, int state);
    
    // import model
    bool import(std::string path);
    
    // output the model
    void output(std::string path);
    
};

#endif /* HMMModelGMM_hpp */
