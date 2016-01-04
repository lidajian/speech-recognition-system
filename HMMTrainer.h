//
//  HMMTrainer.h
//  speech3
//
//  Created by Dajian on 15/12/24.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#ifndef HMMTrainer_h
#define HMMTrainer_h

class HMMTrainer{
    
protected:
    
    int MAX_ITERATIONS; // maximum # of iteration
    
    float MIN_DELTA; // convergence threshold
    
public:
    
    HMMTrainer(){
        MAX_ITERATIONS = 100;
        MIN_DELTA = 1e-3;
    }
    
    // set maximum # of iteration
    void setMaxIteration(int max_iteration){
        MAX_ITERATIONS = max_iteration;
    }
    
    // set convergence threshold
    void setMinDelta(int min_delta){
        MIN_DELTA = min_delta;
    }
    
    // train HMM
    virtual void commit(const std::vector<std::vector<MFCC_Feature> > &) = 0;
    
};


#endif /* HMMTrainer_h */
