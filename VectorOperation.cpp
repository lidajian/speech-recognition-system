//
//  VectorOperation.cpp
//  speech3
//
//  Created by Dajian on 15/12/24.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#include <stdio.h>
#include "VectorOperation.hpp"


MFCC_Feature & addBToA(MFCC_Feature & A, const MFCC_Feature  & B){
    long cursor = A.size() - 1;
    while (cursor >= 0) {
        A[cursor] += B[cursor];
        cursor--;
    }
    return A;
}

MFCC_Feature & divideAByB(MFCC_Feature & A, long B){
    long cursor = A.size() - 1;
    while (cursor >= 0) {
        A[cursor] = A[cursor] / B;
        cursor--;
    }
    return A;
}

float norm(std::vector<float> &A){
    long cursor = A.size() - 1;
    float sum = 0;
    while (cursor >= 0) {
        sum += powf(A[cursor], 2);
        cursor--;
    }
    return sum;
}

// P(xi | thetaj)
double LogGaussianProbability(const MFCC_Feature & xi, MFCC_Feature & miul, std::vector<float> & covl){
    double logp = 0;
    for (int i = 0; i < xi.size(); i++) {
        logp = logp - 0.5 * log(2 * PI * covl[i]) - 0.5 * pow(xi[i]-miul[i],2) / covl[i];
    }
    if (isinf(logp) || logp < MIN_LOG_PROB) {
        return MIN_LOG_PROB;
    }
    return logp;
}

float EuclideanDistance(const MFCC_Feature &A, const MFCC_Feature &B){
    int sum = 0;
    long cursor = A.size() - 1;
    while (cursor >= 0) {
        sum += powf(A[cursor] - B[cursor], 2);
        cursor--;
    }
    return sum;
}

float Log_AddAtoBf(float A, float B){
    float res;
    if (A - B > 10) {
        res = A;
    } else {
        res = B + logf(1 + expf(A - B));
    }
    return res;
}

double Log_AddAtoB(double A, double B){
    if (A - B > 10) {
        return A;
    } else {
        return B + log(1 + exp(A - B));
    }
}

