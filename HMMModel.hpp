//
//  HMMModel.hpp
//  speech3
//
//  Created by Dajian on 15/10/14.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#ifndef HMMModel_hpp
#define HMMModel_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include "Constants.h"
#include "VectorOperation.hpp"

class HMMModel{
    
protected:
    
    int K; // K states
    
    int FEATURE_LENGTH; // # of MFCC features
    
    std::vector<float> * entry_cost; // entry costs
    
    std::vector<std::vector<float> > * transition_cost; // transition costs
    
public:
    
    virtual ~HMMModel(){};
    
    // get # of states
    int getNumberOfStates(){
        return K;
    }
    
    // transition cost from state s1 to state s2
    float getCost(int s1, int s2){
        if (s1==0) {
            return (*entry_cost)[s2-1];
        }else{
            return (*transition_cost)[s1-1][s2-1];
        }
    }
    
    // log( P(v1 | state) )
    virtual float probabilityMFCC(const std::vector<float> & v1, int state, int kernel = 0) = 0;
    
    // Gaussian probability distance function
    virtual float distanceMFCC(const std::vector<float> & v1, int state) = 0;
    
    // import model
    virtual bool import(std::string path) = 0;

};

#endif /* HMMModel_hpp */
