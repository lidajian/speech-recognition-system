//
//  HMMModelSGM.hpp
//  speech3
//
//  Created by Dajian on 15/10/22.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#ifndef HMMModelSGM_hpp
#define HMMModelSGM_hpp

#include "HMMModel.hpp"

class HMMModelSGM: public HMMModel{
    
private:
    
    void clear();

protected:
    
    std::vector<MFCC_Feature> * miu; // centers: K by FEATURE_LENGTH Matrix
    
    std::vector<std::vector<float> > * cov; // covariances: K by FEATURE_LENGTH Matrix

public:
    
    HMMModelSGM();
    
    ~HMMModelSGM();
    
    // log( P(v1 | state) )
    float probabilityMFCC(const std::vector<float> & v1, int state, int kernel = -1);
    
    // Gaussian probability distance function
    float distanceMFCC(const std::vector<float> & v1, int state);
    
    // import model
    bool import(std::string path);
    
    // output the model
    void output(std::string path);
    
};

#endif /* HMMModelSGM_hpp */
