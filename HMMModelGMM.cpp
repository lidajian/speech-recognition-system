//
//  HMMModelGMM.cpp
//  speech3
//
//  Created by Dajian on 15/10/22.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#include "HMMModelGMM.hpp"
#include <math.h>
#include <fstream>

HMMModelGMM::HMMModelGMM(){
    K = 5;
    FEATURE_LENGTH = 39;
    KERNEL_NUMBER = 4;
    miu = NULL;
    cov = NULL;
    transition_cost = NULL;
    entry_cost = NULL;
    alpha = NULL;
}

HMMModelGMM::~HMMModelGMM(){
    clear();
}

void HMMModelGMM::clear(){
    if (miu) {
        delete miu;
        miu = NULL;
    }
    if (cov) {
        delete cov;
        cov = NULL;
    }
    if (entry_cost) {
        delete entry_cost;
        entry_cost = NULL;
    }
    if (transition_cost) {
        delete transition_cost;
        transition_cost = NULL;
    }
    if (alpha) {
        delete alpha;
        alpha = NULL;
    }
}

float HMMModelGMM::probabilityMFCC(const std::vector<float> & v1, int state, int kernel){
    // state range: 1-
    // kernel range: 1-
    std::vector<MFCC_Feature> & vmiu = (*miu)[state-1];
    std::vector<std::vector<float> > & vcov = (*cov)[state-1];
    std::vector<float> & valpha = (*alpha)[state-1];
    if (kernel == 0) {
        float temp_float,temp_sum = valpha[0] + LogGaussianProbability(v1, vmiu[0], vcov[0]);
        for (int i = 1; i < KERNEL_NUMBER; i++) {
            temp_float = valpha[i] + LogGaussianProbability(v1, vmiu[i], vcov[i]);
            temp_sum = Log_AddAtoB(temp_float, temp_sum);
        }
        return temp_sum;
    } else {
        return valpha[kernel-1] + LogGaussianProbability(v1, vmiu[kernel-1], vcov[kernel-1]);
    }
    
}

float HMMModelGMM::distanceMFCC(const MFCC_Feature & v1, int state){
    // state range: 1-
    return -probabilityMFCC(v1, state);
}

bool HMMModelGMM::import(std::string path){
    std::ifstream ifs(path.c_str());
    
    std::string label, stringvalue;
    
    float floatvalue;
    
    int i, j, k;
    
    while (ifs>>label) {
        if (label.compare("[TYPE]") == 0) {
            ifs>>stringvalue;
            if (stringvalue.compare("GMM") != 0) {
                printf("%s is not a Single Gaussian Model!", path.c_str());
                return false;
            }
        }else if (label.compare("[K]") == 0){
            ifs>>K;
        }else if (label.compare("[FEATURE_LENGTH]") == 0){
            ifs>>FEATURE_LENGTH;
        }else if (label.compare("[KERNEL_NUMBER]") == 0){
            ifs>>KERNEL_NUMBER;
        }else if (label.compare("[ENTRY_COST]") == 0){
            entry_cost = new std::vector<float>();
            for (i = 0; i < K; i++) {
                ifs>>floatvalue;
                entry_cost->push_back(floatvalue);
            }
        }else if (label.compare("[TRANSITION_COST]") == 0){
            transition_cost = new std::vector<std::vector<float> >();
            for (i = 0; i < K; i++) {
                MFCC_Feature * temp = new std::vector<float>();
                for (j = 0; j < K; j++) {
                    ifs>>floatvalue;
                    temp->push_back(floatvalue);
                }
                transition_cost->push_back(*temp);
            }
        }else if (label.compare("[MIU]") == 0){
            miu = new std::vector<std::vector<MFCC_Feature> >();
            for (i = 0; i < K; i++) {
                std::vector<MFCC_Feature> * temp1 = new std::vector<MFCC_Feature>();
                for (j = 0; j < KERNEL_NUMBER; j++) {
                    MFCC_Feature * temp2 = new MFCC_Feature();
                    for (k = 0; k < FEATURE_LENGTH; k++) {
                        ifs>>floatvalue;
                        temp2->push_back(floatvalue);
                    }
                    temp1->push_back(*temp2);
                }
                miu->push_back(*temp1);
            }
        }else if (label.compare("[COV]") == 0){
            cov = new std::vector<std::vector<MFCC_Feature> >();
            for (i = 0; i < K; i++) {
                std::vector<MFCC_Feature> * temp1 = new std::vector<MFCC_Feature>();
                for (j = 0; j < KERNEL_NUMBER; j++) {
                    MFCC_Feature * temp2 = new MFCC_Feature();
                    for (k = 0; k < FEATURE_LENGTH; k++) {
                        ifs>>floatvalue;
                        temp2->push_back(floatvalue);
                    }
                    temp1->push_back(*temp2);
                }
                cov->push_back(*temp1);
            }
        }else if (label.compare("[ALPHA]") == 0){
            alpha = new std::vector<std::vector<float> >();
            for (i = 0; i < K; i++) {
                std::vector<float> * temp = new std::vector<float>();
                for (j = 0; j < KERNEL_NUMBER; j++) {
                    ifs>>floatvalue;
                    temp->push_back(floatvalue);
                }
                alpha->push_back(*temp);
            }
        }
    }
    
    ifs.close();
    
    if (cov && miu && transition_cost && entry_cost && alpha) {
        return true;
    }else{
        clear();
        printf("%s is not a complete Gaussian Model file!", path.c_str());
        return false;
    }
}

void HMMModelGMM::output(std::string path){
    if (!miu) {
        return;
    }
    
    printf("Output model!\n");
    
    int i,j,k;
    
    std::ofstream ofs;
    
    ofs.open(path.c_str());
    
    ofs<<"[TYPE]\nGMM\n";
    
    ofs<<"[K]\n"<<K<<"\n";
    
    ofs<<"[FEATURE_LENGTH]\n"<<FEATURE_LENGTH<<"\n";
    
    ofs<<"[KERNEL_NUMBER]\n"<<KERNEL_NUMBER<<"\n";
    
    ofs<<"[ENTRY_COST]\n";
    
    for (i = 0; i < K; i++) {
        ofs<<(*entry_cost)[i]<<" ";
    }
    
    ofs<<"\n";
    
    ofs<<"[TRANSITION_COST]\n";
    
    for (i = 0; i < K; i++) {
        for (j = 0; j < K; j++) {
            ofs<<(*transition_cost)[i][j]<<" ";
        }
        ofs<<"\n";
    }
    
    ofs<<"[MIU]\n";
    
    for (i = 0; i < K; i++) {
        for (j = 0; j < KERNEL_NUMBER; j++) {
            for (k = 0; k < FEATURE_LENGTH; k++) {
                ofs<<(*miu)[i][j][k]<<" ";
            }
            ofs<<"\n";
        }
        ofs<<"\n";
    }
    
    ofs<<"[COV]\n";
    
    for (i = 0; i < K; i++) {
        for (j = 0; j < KERNEL_NUMBER; j++) {
            for (k = 0; k < FEATURE_LENGTH; k++) {
                ofs<<(*cov)[i][j][k]<<" ";
            }
            ofs<<"\n";
        }
        ofs<<"\n";
    }
    
    ofs<<"[ALPHA]\n";
    
    for (i = 0; i < K; i++) {
        for (j = 0; j < KERNEL_NUMBER; j++) {
            ofs<<(*alpha)[i][j]<<" ";
        }
        ofs<<"\n";
    }
    
    ofs.flush();
    
    ofs.close();
    
}
