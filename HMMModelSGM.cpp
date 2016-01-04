//
//  HMMModelSGM.cpp
//  speech3
//
//  Created by Dajian on 15/10/22.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#include "HMMModelSGM.hpp"
#include <math.h>
#include <fstream>

HMMModelSGM::HMMModelSGM(){
    K = 1;
    FEATURE_LENGTH = 1;
    miu = NULL;
    cov = NULL;
    transition_cost = NULL;
    entry_cost = NULL;
}

HMMModelSGM::~HMMModelSGM(){
    clear();
}

void HMMModelSGM::clear(){
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
}

float HMMModelSGM::probabilityMFCC(const std::vector<float> & v1, int state, int kernel){
    return LogGaussianProbability(v1, (*miu)[state-1], (*cov)[state-1]);
}

float HMMModelSGM::distanceMFCC(const MFCC_Feature & v1, int state){
    // state range: 1-
    return -LogGaussianProbability(v1, (*miu)[state-1], (*cov)[state-1]);
}

bool HMMModelSGM::import(std::string path){
    std::ifstream ifs(path.c_str());
    
    std::string label, stringvalue;
    
    float floatvalue;
    
    int i, j;
    
    while (ifs>>label) {
        if (label.compare("[TYPE]") == 0) {
            ifs>>stringvalue;
            if (stringvalue.compare("SGM") != 0) {
                printf("%s is not a Single Gaussian Model!", path.c_str());
                return false;
            }
        }else if (label.compare("[K]") == 0){
            ifs>>K;
        }else if (label.compare("[FEATURE_LENGTH]") == 0){
            ifs>>FEATURE_LENGTH;
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
            miu = new std::vector<MFCC_Feature>();
            for (i = 0; i < K; i++) {
                MFCC_Feature * temp = new std::vector<float>();
                for (j = 0; j < FEATURE_LENGTH; j++) {
                    ifs>>floatvalue;
                    temp->push_back(floatvalue);
                }
                miu->push_back(*temp);
            }
        }else if (label.compare("[COV]") == 0){
            cov = new std::vector<MFCC_Feature>();
            for (i = 0; i < K; i++) {
                MFCC_Feature * temp = new std::vector<float>();
                for (j = 0; j < FEATURE_LENGTH; j++) {
                    ifs>>floatvalue;
                    temp->push_back(floatvalue);
                }
                cov->push_back(*temp);
            }
        }
    }
    
    ifs.close();
    
    if (cov && miu && transition_cost && entry_cost) {
        return true;
    }else{
        clear();
        printf("%s is not a complete Single Gaussian Model file!", path.c_str());
        return false;
    }
}

void HMMModelSGM::output(std::string path){
    if (!miu) {
        return;
    }
    
    printf("Output model!\n");
    
    int i,j;
    
    std::ofstream ofs(path.c_str());
    
    ofs<<"[TYPE]\nSGM\n";
    
    ofs<<"[K]\n"<<K<<"\n";
    
    ofs<<"[FEATURE_LENGTH]\n"<<FEATURE_LENGTH<<"\n";
    
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
        for (j = 0; j < FEATURE_LENGTH; j++) {
            ofs<<(*miu)[i][j]<<" ";
        }
        ofs<<"\n";
    }
    
    ofs<<"[COV]\n";
    
    for (i = 0; i < K; i++) {
        for (j = 0; j < FEATURE_LENGTH; j++) {
            ofs<<(*cov)[i][j]<<" ";
        }
        ofs<<"\n";
    }
    
    ofs.close();
    
}