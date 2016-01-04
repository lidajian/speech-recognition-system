//
//  VectorOperation.hpp
//  speech3
//
//  Created by Dajian on 15/12/24.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#ifndef VectorOperation_h
#define VectorOperation_h

#include <vector>
#include <math.h>
#include "Constants.h"

MFCC_Feature & addBToA(MFCC_Feature & A, const MFCC_Feature  & B);

MFCC_Feature & divideAByB(MFCC_Feature & A, long B);

float norm(std::vector<float> &A);

double LogGaussianProbability(const MFCC_Feature & xi, MFCC_Feature & miul, std::vector<float> & covl);

float EuclideanDistance(const MFCC_Feature &A, const MFCC_Feature &B);

float Log_AddAtoBf(float A, float B);

double Log_AddAtoB(double A, double B);

#endif /* VectorOperation_h */
