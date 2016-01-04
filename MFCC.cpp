//
//  MFCC.cpp
//  speech3
//
//  Created by Dajian on 15/11/24.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#include "MFCC.hpp"

MFCCHandler::MFCCHandler(){
    
    int i, j;
    
    // initialize hamming window
    for(i = 0; i < MFCC_WIDTH; i++)
    {
        hamming_window[i] = 0.54-0.46*cos(2*PI*i/MFCC_WIDTH);
    }
    
    // initialize mel coefficients
    float maxMelFrequency = 1127 * log(1 + MAX_MEL_IN_HZ / 700);
    float deltaMelFrequency = maxMelFrequency / (1 + NUM_FILTER);
    float maxHzFrequency = SAMPLE_RATE / 2;
    float hzfrequencies[NUM_FILTER + 2];
    float temp = 0;
    hzfrequencies[NUM_FILTER + 1] = MAX_MEL_IN_HZ;
    
    for (i = 0; i <= NUM_FILTER; i++) {
        hzfrequencies[i] = 700 * (expf(temp / 1127) - 1);
        temp += deltaMelFrequency;
    }
    
    float deltaHzFrequency = maxHzFrequency / (UNIQ_FFT_COMPONENT - 1);
    for (i = 0; i < NUM_FILTER; i++) {
        float this_freq = 0;
        float left_freq = hzfrequencies[i];
        float mid_freq = hzfrequencies[i+1];
        float right_freq = hzfrequencies[i+2];
        for (j = 0; j < UNIQ_FFT_COMPONENT; j++) {
            if (this_freq <= left_freq) {
                mel_coef[i][j] = 0;
            } else if (this_freq <= mid_freq){
                mel_coef[i][j] = (this_freq - left_freq) / (mid_freq - left_freq);
            } else if (this_freq < right_freq){
                mel_coef[i][j] = (right_freq - this_freq) / (right_freq - mid_freq);
            } else{
                mel_coef[i][j] = 0;
            }
            this_freq += deltaHzFrequency;
        }
    }
    
    // initialize DCT coefficients
    
    for (i = 0; i < NUM_DCT_COEFFICIENT; i++) {
        for (j = 0; j < NUM_FILTER; j++) {
            if (j == 0) {
                dct_coef[i][j] = sqrtf(1.0 / NUM_FILTER) * cosf((j + 0.5) * i * PI / NUM_FILTER);
            }else{
                dct_coef[i][j] = sqrtf(2.0 / NUM_FILTER) * cosf((j + 0.5) * i * PI / NUM_FILTER);
            }
        }
    }
    
}

void MFCCHandler::getMFCCFeature(std::vector<MFCC_Feature> & result, RawRecordData * data){
    
    int i = 0, j = 0, cursor = 0;
    
    if (data->start < 0) {
        data->start = 0;
    }
    
    int dataL = data->end - data->start;
    
    // sample after adding window
    double sample[FFT_ORDER];
    float fftuniq[UNIQ_FFT_COMPONENT];
    float melfiltervalue[NUM_FILTER];
    
    // pad zero
    for (i = MFCC_WIDTH; i < FFT_ORDER; i++) {
        sample[i] = 0;
    }
    
    // pre-emphasis
    float preemp[dataL];
    preemp[0]=data->data[0];
    for(i=1; i<dataL; i++){
        preemp[i] = data->data[i]-0.95*data->data[i-1];
    }
    
    int start = data->start;
    int end = start + MFCC_WIDTH;
    
    std::vector<std::vector<float> > alldct;
    
    while (end < data->end) {
        
        cursor = 0;
        
        // get this sample - result: sample
        for (i = start; i < end; i++) {
            sample[cursor] = data->data[i] * hamming_window[cursor];
            cursor++;
        }
        
        // perform FFT - result: fftuniq
        
        fftw_complex * out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FFT_ORDER);
        fftw_plan p = fftw_plan_dft_r2c_1d(FFT_ORDER, sample, out, FFTW_ESTIMATE);
        fftw_execute(p);
        for (i = 0; i < UNIQ_FFT_COMPONENT; i++) {
            fftuniq[i] = out[i][0] * out[i][0] + out[i][1] * out[i][1];
        }
        fftw_destroy_plan(p);
        fftw_free(out);
        
        
        
        // mel filter - result: melfiltervalue
        
        for (i = 0; i < NUM_FILTER; i++) {
            melfiltervalue[i] = 0;
            for (j = 0; j < UNIQ_FFT_COMPONENT; j++) {
                melfiltervalue[i] += mel_coef[i][j] * fftuniq[j];
            }
            if (melfiltervalue[i] != 0) {
                melfiltervalue[i] = log(melfiltervalue[i]);
            } else {
                melfiltervalue[i] = MIN_LOG_PROB;
            }
            
        }
        
        std::vector<float> dct_result(NUM_DCT_COEFFICIENT, 0);
        
        // DCT
        for (i = 0; i < NUM_DCT_COEFFICIENT; i++) {
            for (j = 0; j < NUM_FILTER; j++) {
                dct_result[i] += dct_coef[i][j] * melfiltervalue[j];
            }
        }
        
        alldct.push_back(dct_result);
        
        start += MFCC_STEP;
        end += MFCC_STEP;
    }
    
    int length = (int)alldct.size();
    
    result.resize(length - 4);
    
    std::vector<double> sum(MFCC_FEATURE_LENGTH, 0);
    std::vector<double> mean(MFCC_FEATURE_LENGTH, 0);
    
    for (i = 2; i < length - 2; i++) {
        std::vector<float> & refresult = result[i - 2];
        refresult.resize(MFCC_FEATURE_LENGTH);
        for (j = 0; j < NUM_DCT_COEFFICIENT; j++) {
            refresult[j] = alldct[i][j];
            sum[j] += refresult[j];
            refresult[j + NUM_DCT_COEFFICIENT] = alldct[i + 1][j] - alldct[i - 1][j];
            sum[j + NUM_DCT_COEFFICIENT] += refresult[j + NUM_DCT_COEFFICIENT];
            refresult[j + 2 * NUM_DCT_COEFFICIENT] = alldct[i + 2][j] - alldct[i - 2][j];
            sum[j + 2 * NUM_DCT_COEFFICIENT] += refresult[j + 2 * NUM_DCT_COEFFICIENT];
        }
    }
    
    // mean and covariance normalization
    
    length -= 4;
    
    for (i = 0; i < MFCC_FEATURE_LENGTH; i++) {
        mean[i] = sum[i] / length;
        sum[i] = 0;
    }
    
    for (i = 0; i < length; i++) {
        std::vector<float> & refresult = result[i];
        for (j = 0; j < MFCC_FEATURE_LENGTH; j++) {
            refresult[j] -= mean[j];
            sum[j] += refresult[j] * refresult[j];
        }
    }
    
    for (i = 0; i < MFCC_FEATURE_LENGTH; i++) {
        sum[i] = sqrt(sum[i] / length);
    }
    
    for (i = 0; i < length; i++) {
        std::vector<float> & refresult = result[i];
        for (j = 0; j < MFCC_FEATURE_LENGTH; j++) {
            
            refresult[j] /= sum[j];
            
        }
    }
    
    delete data;
    
}
