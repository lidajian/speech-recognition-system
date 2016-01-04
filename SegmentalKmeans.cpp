//
//  SegmentalKmeans.cpp
//  speech3
//
//  Created by Dajian on 15/10/9.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#include "SegmentalKmeans.hpp"

void SegmentalKmeans::setParameters(int K, int FEATURE_LENGTH = 39){
    int i;
    
    int temp_num;
    
    this->K = K;
    this->FEATURE_LENGTH = FEATURE_LENGTH;
    
    miu = new std::vector<MFCC_Feature>(K);
    for (i = 0; i < K; i++) {
        MFCC_Feature & temp = (*miu)[i];
        temp.resize(FEATURE_LENGTH, 0);
    }
    
    temp_num = log(K);
    
    cov = new std::vector<std::vector<float> >(K);
    
    entry_cost = new std::vector<float>(K, temp_num);
    
    transition_cost = new std::vector<std::vector<float> >(K);
    
    for (i = 0; i < K; i++) {
        MFCC_Feature & temp = (*transition_cost)[i];
        temp.resize(K, temp_num);
    }
}

void SegmentalKmeans::estimateCovariances(std::vector<std::vector<MFCC_Feature> > &container){
    (*cov).clear();
    long num_frames;
    for (int i = 0; i < K; i++) {
        MFCC_Feature sum(FEATURE_LENGTH, 0);
        std::vector<MFCC_Feature> & temp_ref_K = container[i];
        MFCC_Feature & temp_ref_miu = (*miu)[i];
        num_frames = temp_ref_K.size();
        for (int j = 0; j < num_frames; j++) {
            MFCC_Feature & temp_ref_MFCC = temp_ref_K[j];
            for (int k = 0; k < FEATURE_LENGTH; k++) {
                sum[k] += powf(temp_ref_miu[k] - temp_ref_MFCC[k], 2);
            }
        }
        divideAByB(sum, num_frames);
        (*cov).push_back(sum);
    }
}

void SegmentalKmeans::commit(const std::vector<std::vector<MFCC_Feature> > & samples){
    if (!miu) {
        printf("Please set parameters first.\n");
        return;
    }
    // useful variables
    int i,j,k,min_ind;
    long num_files = samples.size();
    long num_frames;
    int num_iter = 0;
    float temp_float;
    float min;
    int temp_int, temp_int2,cursor;
    std::vector<MFCC_Feature>::const_iterator iter_MFCC;
    MFCC_Feature sum(FEATURE_LENGTH, 0);
    
    // container for each cluster
    std::vector<std::vector<MFCC_Feature> > containers(K); // K | scalable | FEATURE_LENGTH
    
    
    // initialize parameters
    
    // first loop over all frames
    for (i = 0; i < num_files; i++) {
        const std::vector<MFCC_Feature> & temp_ref_vector = samples[i];
        temp_int = ceil(double(samples[i].size()) / K); // # of frames allocated to a state initially
        cursor = 0;
        temp_int2 = 0;
        
        for (iter_MFCC = temp_ref_vector.begin(); iter_MFCC != temp_ref_vector.end(); iter_MFCC++) {
            addBToA((*miu)[cursor], *(iter_MFCC)); // add to miu
            //references: http://stackoverflow.com/questions/27812119/binding-of-reference-to-a-value-of-type-drops-qualifiers
            containers[cursor].push_back(*iter_MFCC); // put into containers
            temp_int2++;
            if (temp_int2==temp_int) {
                cursor++;
                temp_int2 = 0;
            }
        }
    }
    
    // compute miu
    for (i = 0; i < K; i++) {
        divideAByB((*miu)[i], containers[i].size()); // miu initialized!
    }
    
    // compute covariances
    estimateCovariances(containers);
    
    // start iteration
    while (num_iter < MAX_ITERATIONS) {
        // clear containers
        for (i = 0; i < K; i++) {
            containers[i].clear();
        }
        
        // set up a new miu
        std::vector<MFCC_Feature> * new_miu = new std::vector<MFCC_Feature>(K);
        
        for (i = 0; i < K; i++) {
            MFCC_Feature & temp = (*new_miu)[i];
            temp.resize(FEATURE_LENGTH, 0);
        }
        
        // set up state transition matrix
        std::vector<std::vector<int> > transition_matrix(K);
        for (i = 0; i < K; i++) {
            transition_matrix[i].resize(K, 0);
        }
        
        // set up entry vector
        std::vector<int> entry_matrix(K, 0);
        
        // DTW to find the route, put frames into containers, add up containers to miu
        for (i = 0; i < num_files; i++) {
            
            // DTW
            
            const std::vector<MFCC_Feature> & temp_ref_vector = samples[i];
            
            num_frames = temp_ref_vector.size();
            std::vector<std::vector<float> > trellis(K + 1); // DTW trellis
            std::vector<std::vector<int> > next(K + 1); // mark the next state to found
            
            for (j = 0; j <= K; j++) {
                next[j].resize(num_frames + 1, 0);
            }
            
            // first column
            trellis[0].push_back(0);
            next[0][0] = -1;
            for (k = 1; k <= K; k++) {
                trellis[k].push_back(INFINITY);
            }
            
            // other columns
            for (j = 0; j < num_frames; j++) {
                // first rows
                trellis[0].push_back(INFINITY);
                // other rows
                for (k = 1; k <= K; k++) {
                    if (trellis[k][j] == INFINITY) {
                        min = INFINITY;
                    }else{
                        min = trellis[k][j] + (*transition_cost)[k-1][k-1];
                    }
                    min_ind = k;
                    
                    if (trellis[k-1][j] == INFINITY) {
                        temp_float = INFINITY;
                    }else if (k >= 2) {
                        temp_float = trellis[k-1][j] + (*transition_cost)[k-2][k-1];
                    }else{
                        temp_float = trellis[k-1][j] + (*entry_cost)[k-1];
                    }
                    
                    if (temp_float < min) {
                        min = temp_float;
                        min_ind = k - 1;
                    }
                    
                    if (k >= 2) {
                        if (trellis[k-2][j] == INFINITY) {
                            temp_float = INFINITY;
                        }else if (k >= 3) {
                            temp_float = trellis[k-2][j] + (*transition_cost)[k-3][k-1];
                        }else{
                            temp_float = trellis[k-2][j] + (*entry_cost)[k-1];
                        }
                        
                        if (temp_float < min) {
                            min = temp_float;
                            min_ind = k - 2;
                        }
                    }
                    
                    trellis[k].push_back(min + distanceMFCC(temp_ref_vector[j], k));
                    next[k][j + 1] = min_ind;
                }
            }
            
            // find the route stored in 'next'
            long cursor_frame = num_frames;
            int cursor_state = K;
            while (next[cursor_state][cursor_frame] != -1) {
                containers[cursor_state - 1].push_back(temp_ref_vector[cursor_frame-1]); // put into containers
                addBToA((*new_miu)[cursor_state - 1], temp_ref_vector[cursor_frame-1]);
                int next_cursor_state = next[cursor_state][cursor_frame];
                if (next_cursor_state == 0) {
                    entry_matrix[cursor_state-1]++; // add to entry matrix
                }else{
                    transition_matrix[next_cursor_state-1][cursor_state-1]++; // add to transition matrix
                }
                cursor_state = next_cursor_state;
                cursor_frame--;
            }
            
        }
        
        // re-estimate transition cost
        for (i = 0; i < K; i++) {
            for (j = 0; j < K; j++) {
                if (transition_matrix[i][j] == 0) {
                    (*transition_cost)[i][j] = MAX_IN_DTW;
                }else{
                    (*transition_cost)[i][j] = -log((float)transition_matrix[i][j]  / containers[i].size());
                }
                
            }
        }
        
        // re-estimate entry cost
        for (i = 0; i < K; i++) {
            if (entry_matrix[i] == 0) {
                (*entry_cost)[i] = MAX_IN_DTW;
            }else{
                (*entry_cost)[i] = -log(float(entry_matrix[i]) / num_files);
            }
        }
        
        // re-estimate miu
        for (i = 0; i < K; i++) {
            divideAByB((*new_miu)[i], containers[i].size());
        }
        
        // calculate delta(miu)
        temp_float = 0;
        for (i = 0; i < K; i++) {
            for (j = 0; j < FEATURE_LENGTH; j++) {
                temp_float += powf((*miu)[i][j] - (*new_miu)[i][j], 2);
            }
        }
        temp_float = sqrtf(temp_float);
        
        delete miu;
        miu = new_miu;
        
        // re-estimate covariances
        estimateCovariances(containers);
        
        if (temp_float < MIN_DELTA) {
            break;
        }
        
        num_iter++;
    }
    
    if (num_iter == MAX_ITERATIONS) {
        printf("Fail to converge in %d iterations.\n", num_iter);
    }else{
        printf("Converge in %d iterations.\n", num_iter);
    }
    
}