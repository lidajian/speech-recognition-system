//
//  MFCC.hpp
//  speech3
//
//  Created by Dajian on 15/11/24.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#ifndef MFCC_hpp
#define MFCC_hpp

#include <stdio.h>
#include <vector>
#include <math.h>
#include "fftw3.h"
#include "Constants.h"

class MFCCHandler{
    
private:
    float hamming_window[MFCC_WIDTH];
    float dct_coef[MFCC_FEATURE_LENGTH / 3][NUM_FILTER];
    float mel_coef[NUM_FILTER][UNIQ_FFT_COMPONENT];
    
public:
    // constructor of MFCCHandler, prepare common coefficients for calculating MFCC feature
    MFCCHandler();
    
    // calculate MFCC feature
    void getMFCCFeature(std::vector<MFCC_Feature> & result, RawRecordData * data);
    
};


#endif /* MFCC_hpp */
