//
//  BaumWelchHMM.cpp
//  speech3
//
//  Created by Dajian on 15/12/23.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#include "BaumWelchHMM.hpp"
#include "DTWGallery.hpp"

void BaumWelchHMM::split_kmeans(std::vector<std::vector<float> > &miu_one_state, std::vector<std::vector<float> > &cov_one_state, std::vector<std::vector<float> > &container){
    
    int i, j, k, max_ind, min_ind;
    float max_v, min_v, temp_float;
    long num_kernel;
    
    max_ind = 0;
    max_v = norm(cov_one_state[0]);
    
    // split
    
    num_kernel = cov_one_state.size();
    
    for (i = 1; i < num_kernel; i++) {
        temp_float = norm(cov_one_state[i]);
        if (max_v < temp_float) {
            max_v = temp_float;
            max_ind = i;
        }
    }
    
    std::vector<float> temp(FEATURE_LENGTH);
    
    for (i = 0; i < FEATURE_LENGTH; i++) {
        temp_float = cov_one_state[max_ind][i] / 4;
        temp[i] = miu_one_state[max_ind][i] - temp_float;
        miu_one_state[max_ind][i] += temp_float;
    }
    
    miu_one_state.push_back(temp);
    cov_one_state.push_back(cov_one_state[max_ind]);
    
    // kmeans
    num_kernel++;
    
    std::vector<std::vector<int> > container_kmeans(num_kernel);
    
    for (int num_iter = 0; num_iter < MAX_ITERATIONS; num_iter++) {
        
        // clear container
        for (i = 0; i < num_kernel; i++) {
            container_kmeans[i].clear();
        }
        
        std::vector<std::vector<float> > new_miu(num_kernel);
        for (i = 0; i < num_kernel; i++) {
            new_miu[i].resize(FEATURE_LENGTH, 0);
        }
        
        for (i = (int)container.size() - 1; i >= 0 ; i--) {
            min_ind = 0;
            min_v = EuclideanDistance(miu_one_state[0], container[i]);
            
            for (j = 1; j < num_kernel; j++) {
                temp_float = EuclideanDistance(miu_one_state[j], container[i]);
                
                if (temp_float < min_v) {
                    min_v = temp_float;
                    min_ind = j;
                }
            }
            
            // add to containerkmeans and new miu
            addBToA(new_miu[min_ind], container[i]);
            container_kmeans[min_ind].push_back(i);
        }
        
        // reestimate miu
        for (i = 0; i < num_kernel; i++) {
            divideAByB(new_miu[i], container_kmeans[i].size());
        }
        
        temp_float = 0;
        for (i = 0; i < num_kernel; i++) {
            temp_float += EuclideanDistance(new_miu[i], miu_one_state[i]);
        }
        
        for (i = 0; i < num_kernel; i++) {
            miu_one_state[i] = new_miu[i];
        }
        
        if (temp_float < MIN_DELTA) {
            break;
        }
        
    }
    
    // estimate covariance
    for (i = 0; i < num_kernel; i++) {
        long num_frames = container_kmeans[i].size();
        for (j = 0; j < FEATURE_LENGTH; j++) {
            if (num_frames == 1) {
                cov_one_state[i][j] = MCOV;
            }else{
                cov_one_state[i][j] = 0;
                for (k = 0; k < num_frames; k++) {
                    cov_one_state[i][j] += powf(container[container_kmeans[i][k]][j] - miu_one_state[i][j], 2);
                }
            }
        }
        divideAByB(cov_one_state[i], num_frames);
    }
    
}

void BaumWelchHMM::setParameters(int K, int FEATURE_LENGTH, int KERNEL_NUMBER){
    int i = 0;
    float temp_float = 0;
    this->K = K;
    this->FEATURE_LENGTH = FEATURE_LENGTH;
    this->KERNEL_NUMBER = KERNEL_NUMBER;

    // initialize entry_cost and transition_cost
    temp_float = log(K);
    
    entry_cost = new std::vector<float>(K, temp_float);
    
    transition_cost = new std::vector<std::vector<float> >(K);
    
    for (i = 0; i < K; i++) {
        MFCC_Feature & temp = (*transition_cost)[i];
        temp.resize(K, temp_float);
    }
     
    // initialize alpha
    temp_float = -log (KERNEL_NUMBER);
    alpha = new std::vector<std::vector<float> >(K);
    for (i = 0; i < K; i++) {
        std::vector<float> & ref_alpha_1 = (*alpha)[i];
        ref_alpha_1.resize(KERNEL_NUMBER, temp_float);
    }

    
    // NOTICE: miu, cov are NOT initialized!
    
}

void BaumWelchHMM::initializeGMMParameters(std::vector<std::vector<MFCC_Feature> > & container){
    
    if (miu) {
        delete miu;
        delete cov;
    }
    
    miu = new std::vector<std::vector<MFCC_Feature> >(K);
    cov = new std::vector<std::vector<std::vector<float> > >(K);
    
    for (int i = 0; i < K; i++) {
        
        // one big cluster for each state
        std::vector<float> temp(FEATURE_LENGTH, 0);
        
        unsigned long num_frames = container[i].size();
        
        for (int j = 0; j < num_frames; j++) {
            addBToA(temp, container[i][j]);
        }
        
        divideAByB(temp, num_frames);
        
        (*miu)[i].push_back(temp);
        (*cov)[i].push_back(temp);
        
        for (int j = 1; j < KERNEL_NUMBER; j++) {
            split_kmeans((*miu)[i], (*cov)[i], container[i]);
        }
        
    }
    
    
}

float BaumWelchHMM::estimateGMM(const std::vector<std::vector<MFCC_Feature> > & samples,
                 // for soft calculate transition probability
                 std::vector<std::vector<std::vector<std::vector<float> > > > & buffer_alphas,
                 std::vector<std::vector<std::vector<std::vector<float> > > > & buffer_betas,
                 std::vector<std::vector<std::vector<std::vector<float> > > > & buffer_pxs,
                 // for soft calculate miu, cov, alpha
                 std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > & buffer_lambdas,
                 // the index of the model, for extract required data to calculate parameters
                 int index_Model
                 ){
    
    int i,j,k,l,m;
    float temp_lambda,temp_float,sum_float;
    
    float ret = 0; // returned value(for convergence determination): distance between old miu and new miu
    
    
    long num_samples = samples.size();
    
    // update miu, cov, alpha
    std::vector<std::vector<float> > sum_lambda(K);
    std::vector<std::vector<MFCC_Feature> > sum_miu(K);
    std::vector<std::vector<MFCC_Feature> > sum_cov(K);
    
    for (i = 0; i < K; i++) {
        sum_cov[i].resize(KERNEL_NUMBER);
        sum_miu[i].resize(KERNEL_NUMBER);
        sum_lambda[i].resize(KERNEL_NUMBER, MIN_LOG_PROB);
        for (j = 0; j < KERNEL_NUMBER; j++) {
            sum_miu[i][j].resize(FEATURE_LENGTH, 0);
            sum_cov[i][j].resize(FEATURE_LENGTH, MIN_LOG_PROB);
        }
    }
    
    // on lambda, first run - sum_lambda & sum_miu
    for (i = 0; i < num_samples; i++) {
        
        const std::vector<std::vector<std::vector<float> > > & ref_lambda = buffer_lambdas[i][index_Model];
        
        if (ref_lambda.size() != 0) {
            long num_frames = samples[i].size();
            
            for (j = 0; j < K; j++) {
                for (k = 0; k < KERNEL_NUMBER; k++) {
                    for (l = 0; l < num_frames; l++) {
                        
                        const MFCC_Feature & frame = samples[i][l];
                        
                        temp_lambda = ref_lambda[j][k][l];
                        sum_lambda[j][k] = Log_AddAtoBf(temp_lambda, sum_lambda[j][k]);
                        for (m = 0; m < FEATURE_LENGTH; m++) {
                            if (frame[m] != 0 && temp_lambda > -100) {
                                temp_float = frame[m] * expf(temp_lambda);
                                sum_miu[j][k][m] += temp_float;
                                
                            }
                            
                        }
                    }
                }
            }
        }
        
    }
    
    for (i = 0; i < K; i++) {
        for (j = 0; j < KERNEL_NUMBER; j++) {
            temp_float = exp(sum_lambda[i][j]);
            for (k = 0; k < FEATURE_LENGTH; k++) {
                ret += ((*miu)[i][j][k] - sum_miu[i][j][k] / temp_float) * ((*miu)[i][j][k] - sum_miu[i][j][k] / temp_float);
                (*miu)[i][j][k] = sum_miu[i][j][k] / temp_float; // update miu HERE
            }
        }
    }
    
    // on lambda, second run - sum_cov
    for (i = 0; i < num_samples; i++) {
        
        const std::vector<std::vector<std::vector<float> > > & ref_lambda = buffer_lambdas[i][index_Model];
        
        if (ref_lambda.size() != 0) {
            long num_frames = samples[i].size();
            
            for (j = 0; j < K; j++) {
                for (k = 0; k < KERNEL_NUMBER; k++) {
                    for (l = 0; l < num_frames; l++) {
                        
                        const MFCC_Feature & frame = samples[i][l];
                        
                        temp_lambda = ref_lambda[j][k][l];
                        for (m = 0; m < FEATURE_LENGTH; m++) {
                            
                            temp_float = fabsf(frame[m] - (*miu)[j][k][m]);
                            
                            if (temp_float != 0) {
                                temp_float = 2 * logf(temp_float) + temp_lambda;
                                sum_cov[j][k][m] = Log_AddAtoBf(temp_float, sum_cov[j][k][m]);
                            }
                            
                        }
                    }
                }
            }
        }
        
    }
    
    for (i = 0; i < K; i++) {
        for (j = 0; j < KERNEL_NUMBER; j++) {
            for (k = 0; k < FEATURE_LENGTH; k++) {
                temp_float = expf(sum_cov[i][j][k] - sum_lambda[i][j]);
                
                if (temp_float == 0) {
                    temp_float = 0.001;
                }
                
                (*cov)[i][j][k] = temp_float; // update cov HERE
            }
        }
    }
    
    for (i = 0; i < K; i++) {
        temp_float = sum_lambda[i][0];
        for (j = 1; j < KERNEL_NUMBER; j++) {
            temp_float = Log_AddAtoBf(sum_lambda[i][j], temp_float);
        }
        for (j = 0; j < KERNEL_NUMBER; j++) {
            (*alpha)[i][j] = sum_lambda[i][j] - temp_float; // update alpha HERE
        }
    }
    
    // on pxs, alpha, beta: find transition costs
    std::vector<float> temp_cost(K, MIN_LOG_PROB);
    std::vector<std::vector<float> > temp_transition_cost(K);
    std::vector<std::vector<float> > new_transition_cost(K);
    
    
    for (i = 0; i < K; i++) {
        temp_transition_cost[i].resize(K);
        new_transition_cost[i].resize(K, MIN_LOG_PROB);
        (*entry_cost)[i] = MIN_LOG_PROB;
    }
    
    // entry cost
    for (i = 0; i < num_samples; i++) {
        
        const std::vector<std::vector<float> > & ref_pxs = buffer_pxs[i][index_Model];
        const std::vector<std::vector<float> > & ref_betas = buffer_betas[i][index_Model];
        
        if (buffer_betas[i][index_Model].size() != 0) {
            sum_float = MIN_LOG_PROB;
            
            for (j = 1; j <= K; j++) {
                temp_float = ref_pxs[j][0] - getCost(0, j) + ref_betas[j][0];
                temp_cost[j-1] = temp_float;
                sum_float = Log_AddAtoBf(temp_float, sum_float);
            }
            
            for (j = 0; j < K; j++) {
                (*entry_cost)[j] = Log_AddAtoBf(temp_cost[j] - sum_float, (*entry_cost)[j]);
            }
        }
        
    }
    
    sum_float = MIN_LOG_PROB;
    
    for (i = 0; i < K; i++) {
        sum_float = Log_AddAtoBf((*entry_cost)[i], sum_float);
    }
    for (i = 0; i < K; i++) {
        (*entry_cost)[i] = sum_float - (*entry_cost)[i]; // update entry cost HERE
    }
    
    // transition cost
    for (i = 0; i < num_samples; i++) {
        
        const std::vector<std::vector<float> > & ref_pxs = buffer_pxs[i][index_Model];
        const std::vector<std::vector<float> > & ref_alphas = buffer_alphas[i][index_Model];
        const std::vector<std::vector<float> > & ref_betas = buffer_betas[i][index_Model];
        
        if (buffer_betas[i][index_Model].size() != 0) {
        
            for (j = (int)samples[i].size() - 2; j >= 0; j--) {
                
                sum_float = MIN_LOG_PROB;
                
                for (k = 1; k <= K; k++) {
                    
                    for (l = k; l <= K; l++) {
                        temp_float = ref_pxs[l][j+1] - getCost(k, l) + ref_alphas[k][j] + ref_betas[l][j+1];
                        sum_float = Log_AddAtoBf(temp_float, sum_float);
                        temp_transition_cost[k-1][l-1] = temp_float;
                    }
                    
                }
                
                for (k = 0; k < K; k++) {
                    for (l = k; l < K; l++) {
                        new_transition_cost[k][l] = Log_AddAtoBf(temp_transition_cost[k][l] - sum_float, new_transition_cost[k][l]);
                    }
                }
                
            }
        }

    }
    
    
    
    for (i = 0; i < K; i++) {
        
        sum_float = MIN_LOG_PROB;
        
        for (j = 0; j < K; j++) {
            sum_float = Log_AddAtoBf(new_transition_cost[i][j], sum_float);
        }
        
        for (j = 0; j < K; j++) {
            (*transition_cost)[i][j] = sum_float - new_transition_cost[i][j]; // update transition cost HERE
        }
        
    }
    
    return sqrtf(ret);
    
}

void BaumWelchHMM::commit(const std::vector<std::vector<MFCC_Feature> > & samples){
    if (!alpha) {
        printf("Please set parameters first.\n");
        return;
    }
    
    // useful variables
    int i;
    long num_files = samples.size();
    int num_iter = 0;
    int temp_int, temp_int2,cursor;
    std::vector<MFCC_Feature>::const_iterator iter_MFCC;
    MFCC_Feature sum(FEATURE_LENGTH, 0);
    
    // container for each cluster
    std::vector<std::vector<MFCC_Feature> > containers(K); // K | scalable | FEATURE_LENGTH
    
    
    // initialize parameters
    
    // first loop over all frames, allocate frames for initialization
    for (i = 0; i < num_files; i++) {
        const std::vector<MFCC_Feature> & temp_ref_vector = samples[i];
        temp_int = ceil(double(samples[i].size()) / K); // # of frames allocated to a state initially
        cursor = 0;
        temp_int2 = 0;
        for (iter_MFCC = temp_ref_vector.begin(); iter_MFCC != temp_ref_vector.end(); iter_MFCC++) {
            containers[cursor].push_back(*iter_MFCC); // put into containers
            temp_int2++;
            if (temp_int2==temp_int) {
                cursor++;
                temp_int2 = 0;
            }
        }
    }
    
    printf("Initializing GMM...\n");
    
    // estimate states distribution
    initializeGMMParameters(containers);
    
    DTWGallery gallery;
    
    std::vector<BaumWelchHMM *> dictionary(1,this);
    
    std::vector<int> order(1,0);
    
    while (num_iter < MAX_ITERATIONS) {
        std::vector<std::vector<std::vector<std::vector<float> > > > buffer_alphas;
        std::vector<std::vector<std::vector<std::vector<float> > > > buffer_betas;
        std::vector<std::vector<std::vector<std::vector<float> > > > buffer_pxs;
        std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > buffer_lambdas;
        
        for (i = 0; i < num_files; i++) {
            gallery.BackwardForward(dictionary, order, samples[i], buffer_alphas, buffer_betas, buffer_pxs, buffer_lambdas);
        }
        
        if (estimateGMM(samples, buffer_alphas, buffer_betas, buffer_pxs, buffer_lambdas) < MIN_DELTA) {
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


