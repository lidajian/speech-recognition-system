//
//  main.cpp
//  speech3
//
//  Created by Dajian on 15/10/6.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include "SegmentalKmeans.hpp"
#include "SegmentalKmeansGMM.hpp"
#include "HMMModel.hpp"
#include "HMMModelGMM.hpp"
#include "HMMModelSGM.hpp"
#include "RecorderHelper.hpp"
#include "BaumWelchHMM.hpp"
#include "fftw3.h"
#include "MFCC.hpp"
#include "DTWGallery.hpp"

#define FEATURE_LENGTH 39

#define STRING_BUFFER_SIZE 200


/*
 -------------------------- train using Baum Welch (on my dataset)
 */

int main(){
    
    int i,j;
    float sum;
    
    MFCCHandler handler;
    DTWGallery gallery;
    
    BaumWelchHMM model0;
    model0.import("/Users/apple/Desktop/mymodel/zero.txt");
    
    BaumWelchHMM model1;
    model1.import("/Users/apple/Desktop/mymodel/one.txt");
    
    BaumWelchHMM model2;
    model2.import("/Users/apple/Desktop/mymodel/two.txt");
    
    BaumWelchHMM model3;
    model3.import("/Users/apple/Desktop/mymodel/three.txt");
    
    BaumWelchHMM model4;
    model4.import("/Users/apple/Desktop/mymodel/four.txt");
    
    BaumWelchHMM model5;
    model5.import("/Users/apple/Desktop/mymodel/five.txt");
    
    BaumWelchHMM model6;
    model6.import("/Users/apple/Desktop/mymodel/six.txt");
    
    BaumWelchHMM model7;
    model7.import("/Users/apple/Desktop/mymodel/seven.txt");
    
    BaumWelchHMM model8;
    model8.import("/Users/apple/Desktop/mymodel/eight.txt");
    
    BaumWelchHMM model9;
    model9.import("/Users/apple/Desktop/mymodel/nine.txt");
    
    BaumWelchHMM model10;
    model10.import("/Users/apple/Desktop/mymodel/sil.txt");
    
    BaumWelchHMM model11;
    model11.import("/Users/apple/Desktop/mymodel/oh.txt");
    
    std::vector<BaumWelchHMM *> dictionary;
    
    dictionary.push_back(&model0);
    dictionary.push_back(&model1);
    dictionary.push_back(&model2);
    dictionary.push_back(&model3);
    dictionary.push_back(&model4);
    dictionary.push_back(&model5);
    dictionary.push_back(&model6);
    dictionary.push_back(&model7);
    dictionary.push_back(&model8);
    dictionary.push_back(&model9);
    dictionary.push_back(&model10);
//    dictionary.push_back(&model11);
    
    std::vector<std::vector<MFCC_Feature>> samples;
    std::vector<std::vector<int>> orders;

    std::ifstream filelist, groundtruth;

    filelist.open("/Users/apple/Desktop/mycontineousaudio/filelist.txt");
    groundtruth.open("/Users/apple/Desktop/mycontineousaudio/labels.txt");

    char wavfilename[STRING_BUFFER_SIZE];

    while (filelist.getline(wavfilename, STRING_BUFFER_SIZE)) {

        char wavfilepath[STRING_BUFFER_SIZE];
        sprintf(wavfilepath, "/Users/apple/Desktop/mycontineousaudio/audio/%s",wavfilename);
        //std::cout<<wavfilepath<<std::endl;

        RawRecordData * data = ReadWaveFile(wavfilepath);

        std::vector<MFCC_Feature> sample;

        handler.getMFCCFeature(sample, data);

        samples.push_back(sample);

        std::string word;

        std::vector<int> order;

        while (groundtruth >> word) {
            if (word == "zero") {
                order.push_back(0);
            } else if (word == "one"){
                order.push_back(1);
            } else if (word == "two"){
                order.push_back(2);
            } else if (word == "three"){
                order.push_back(3);
            } else if (word == "four"){
                order.push_back(4);
            } else if (word == "five"){
                order.push_back(5);
            } else if (word == "six"){
                order.push_back(6);
            } else if (word == "seven"){
                order.push_back(7);
            } else if (word == "eight"){
                order.push_back(8);
            } else if (word == "nine"){
                order.push_back(9);
            } else if (word == "sil"){
                order.push_back(10);
            } else if (word == "oh") {
                order.push_back(11);
            } else{
                break;
            }
        }

        orders.push_back(order);

        std::cout<<wavfilepath<<" ";

        for (i = 0; i < order.size(); i++) {
            std::cout<<order[i];
        }


    }


    filelist.close();
    groundtruth.close();


    std::cout<<"all loaded"<<std::endl;
    // start train
    
    for (i = 0; i < 100; i++) {
        
        std::vector<std::vector<std::vector<std::vector<float> > > > buffer_alphas;
        std::vector<std::vector<std::vector<std::vector<float> > > > buffer_betas;
        std::vector<std::vector<std::vector<std::vector<float> > > > buffer_pxs;
        std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > buffer_lambdas;
        
        sum = 0;
        
        for (j = 0; j < samples.size(); j++) {
            gallery.BackwardForward(dictionary, orders[j], samples[j], buffer_alphas, buffer_betas, buffer_pxs, buffer_lambdas);
        }
        
        for (j = 0; j < 10; j++) {
            sum = dictionary[j]->estimateGMM(samples, buffer_alphas, buffer_betas, buffer_pxs, buffer_lambdas, j);
        }
        
        printf("sum: %e\n", sum);
        
        if (sum < 0.01) {
            break;
        }
        
    }
    
    char outputfilepath[STRING_BUFFER_SIZE];
    
    for (i = 0; i < dictionary.size(); i++) {
        sprintf(outputfilepath, "/Users/apple/Desktop/newmodel/%d.txt",i);
        dictionary[i]->output(outputfilepath);
    }
    
}


/*
 
 ---------------------- train with continuous speech (release)
 
 */

//
//int main(int argc, const char * argv[]) {
//    
//    int i,j,cursor,sum;
//    
//    MFCCHandler handler;
//    
//    DTWGallery gallery;
//    
//    SegmentalKmeansGMM * model0 = new SegmentalKmeansGMM();
//    model0->import("/Users/apple/Desktop/mymodel/zero.txt");
//    
//    SegmentalKmeansGMM * model1 = new SegmentalKmeansGMM();
//    model1->import("/Users/apple/Desktop/mymodel/one.txt");
//    
//    SegmentalKmeansGMM * model2 = new SegmentalKmeansGMM();
//    model2->import("/Users/apple/Desktop/mymodel/two.txt");
//    
//    SegmentalKmeansGMM * model3 = new SegmentalKmeansGMM();
//    model3->import("/Users/apple/Desktop/mymodel/three.txt");
//    
//    SegmentalKmeansGMM * model4 = new SegmentalKmeansGMM();
//    model4->import("/Users/apple/Desktop/mymodel/four.txt");
//    
//    SegmentalKmeansGMM * model5 = new SegmentalKmeansGMM();
//    model5->import("/Users/apple/Desktop/mymodel/five.txt");
//    
//    SegmentalKmeansGMM * model6 = new SegmentalKmeansGMM();
//    model6->import("/Users/apple/Desktop/mymodel/six.txt");
//    
//    SegmentalKmeansGMM * model7 = new SegmentalKmeansGMM();
//    model7->import("/Users/apple/Desktop/mymodel/seven.txt");
//    
//    SegmentalKmeansGMM * model8 = new SegmentalKmeansGMM();
//    model8->import("/Users/apple/Desktop/mymodel/eight.txt");
//    
//    SegmentalKmeansGMM * model9 = new SegmentalKmeansGMM();
//    model9->import("/Users/apple/Desktop/mymodel/nine.txt");
//    
//    SegmentalKmeansGMM * model10 = new SegmentalKmeansGMM();
//    model10->import("/Users/apple/Desktop/mymodel/sil.txt");
//    
//    SegmentalKmeansGMM * model11 = new SegmentalKmeansGMM();
//    model11->import("/Users/apple/Desktop/mymodel/oh.txt");
//    
//    std::vector<SegmentalKmeansGMM *> dictionary;
//    dictionary.push_back(model0);
//    dictionary.push_back(model1);
//    dictionary.push_back(model2);
//    dictionary.push_back(model3);
//    dictionary.push_back(model4);
//    dictionary.push_back(model5);
//    dictionary.push_back(model6);
//    dictionary.push_back(model7);
//    dictionary.push_back(model8);
//    dictionary.push_back(model9);
//    dictionary.push_back(model10);
//    dictionary.push_back(model11);
//    
//    std::vector<std::vector<MFCC_Feature>> samples;
//    std::vector<std::vector<int>> orders;
//    
//    std::ifstream filelist, groundtruth;
//    
//    filelist.open("/Users/apple/Downloads/hwdata/TRAIN.filelist");
//    groundtruth.open("/Users/apple/Downloads/hwdata/TRAIN.transcripts");
//    
//    char wavfilename[STRING_BUFFER_SIZE];
//    
//    while (filelist.getline(wavfilename, STRING_BUFFER_SIZE)) {
//        
//        char wavfilepath[STRING_BUFFER_SIZE];
//        sprintf(wavfilepath, "/Users/apple/Downloads/hwdata/train/%s.wav",wavfilename);
//        //std::cout<<wavfilepath<<std::endl;
//        
//        RawRecordData * data = ReadWaveFile(wavfilepath);
//        
//        std::vector<MFCC_Feature> sample;
//        
//        handler.getMFCCFeature(sample, data);
//        
//        samples.push_back(sample);
//        
//        std::string word;
//        
//        std::vector<int> order;
//        
//        while (groundtruth >> word) {
//            if (word == "zero") {
//                order.push_back(0);
//            } else if (word == "one"){
//                order.push_back(1);
//            } else if (word == "two"){
//                order.push_back(2);
//            } else if (word == "three"){
//                order.push_back(3);
//            } else if (word == "four"){
//                order.push_back(4);
//            } else if (word == "five"){
//                order.push_back(5);
//            } else if (word == "six"){
//                order.push_back(6);
//            } else if (word == "seven"){
//                order.push_back(7);
//            } else if (word == "eight"){
//                order.push_back(8);
//            } else if (word == "nine"){
//                order.push_back(9);
//            } else if (word == "sil"){
//                order.push_back(10);
//            } else if (word == "oh") {
//                order.push_back(11);
//            } else{
//                break;
//            }
//        }
//        
//        orders.push_back(order);
//        
//        std::cout<<wavfilepath<<" ";
//        
//        for (i = 0; i < order.size(); i++) {
//            std::cout<<order[i];
//        }
//        
//        
//    }
//    
//    
//    filelist.close();
//    groundtruth.close();
//    
//    
//    std::cout<<"all loaded"<<std::endl;
//    // start train
//    
//    // convergence
//    int num_states = 0;
//    for (i = 0; i < dictionary.size(); i++) {
//        num_states += dictionary[i]->getNumberOfStates();
//    }
//    
//    int * histgram = new int[num_states];
//    
//    for (i = 0; i < num_states; i++) {
//        histgram[i] = 0;
//    }
//    
//    
//    for (int iter = 0; iter < 100; iter++) {
//        
//        
//        
//        std::vector<std::vector<std::vector<MFCC_Feature>>> containers(dictionary.size());
//        
//        for (i = 0; i < dictionary.size(); i++) {
//            containers[i].resize(dictionary[i]->getNumberOfStates());
//        }
//        
//        for (i = 0; i < samples.size(); i++) {
//            delete gallery.DTW_sequence_structure_configured(dictionary, orders[i], samples[i], containers);
//        }
//        
//        cursor = 0;
//        sum = 0;
//        
//        for (i = (int)dictionary.size() - 1; i >=0 ; i--) {
//            printf("we are at model: %d",i);
//            dictionary[i]->estimateGMM(containers[i]);
//            for (j = 0; j < dictionary[i]->getNumberOfStates(); j++) {
//                sum += abs(histgram[cursor] - (int)containers[i][j].size());
//                histgram[cursor++] = (int)containers[i][j].size();
//            }
//        }
//        
//        std::cout<<"iter:"<<iter<<":"<<sum<<std::endl;
//        
//        if (sum == 0) {
//            break;
//        }
//    }
//    
//    char outputfilepath[STRING_BUFFER_SIZE];
//    
//    for (i = 0; i < dictionary.size(); i++) {
//        sprintf(outputfilepath, "/Users/apple/Desktop/newmodel/%d.txt",i);
//        dictionary[i]->output(outputfilepath);
//    }
//    
//    
//    delete model0;
//    delete model1;
//    delete model2;
//    delete model3;
//    delete model4;
//    delete model5;
//    delete model6;
//    delete model7;
//    delete model8;
//    delete model9;
//    delete model10;
//    delete model11;
//    
//}




/*
 
 ---------------------- train with continuous speech (debug)
 
 */

//int main(int argc, const char * argv[]) {
//    
//    MFCCHandler handler;
//    
//    DTWGallery gallery;
//    
//    SegmentalKmeansGMM * model0 = new SegmentalKmeansGMM();
//    model0->import("/Users/apple/Desktop/mymodel/zero.txt");
//    
//    SegmentalKmeansGMM * model1 = new SegmentalKmeansGMM();
//    model1->import("/Users/apple/Desktop/mymodel/one.txt");
//    
//    SegmentalKmeansGMM * model2 = new SegmentalKmeansGMM();
//    model2->import("/Users/apple/Desktop/mymodel/two.txt");
//    
//    SegmentalKmeansGMM * model3 = new SegmentalKmeansGMM();
//    model3->import("/Users/apple/Desktop/mymodel/three.txt");
//    
//    SegmentalKmeansGMM * model4 = new SegmentalKmeansGMM();
//    model4->import("/Users/apple/Desktop/mymodel/four.txt");
//    
//    SegmentalKmeansGMM * model5 = new SegmentalKmeansGMM();
//    model5->import("/Users/apple/Desktop/mymodel/five.txt");
//    
//    SegmentalKmeansGMM * model6 = new SegmentalKmeansGMM();
//    model6->import("/Users/apple/Desktop/mymodel/six.txt");
//    
//    SegmentalKmeansGMM * model7 = new SegmentalKmeansGMM();
//    model7->import("/Users/apple/Desktop/mymodel/seven.txt");
//    
//    SegmentalKmeansGMM * model8 = new SegmentalKmeansGMM();
//    model8->import("/Users/apple/Desktop/mymodel/eight.txt");
//    
//    SegmentalKmeansGMM * model9 = new SegmentalKmeansGMM();
//    model9->import("/Users/apple/Desktop/mymodel/nine.txt");
//
//    SegmentalKmeansGMM * model10 = new SegmentalKmeansGMM();
//    model10->import("/Users/apple/Desktop/mymodel/sil.txt");
//    
//    std::vector<SegmentalKmeansGMM *> dictionary;
//    dictionary.push_back(model0);
//    dictionary.push_back(model1);
//    dictionary.push_back(model2);
//    dictionary.push_back(model3);
//    dictionary.push_back(model4);
//    dictionary.push_back(model5);
//    dictionary.push_back(model6);
//    dictionary.push_back(model7);
//    dictionary.push_back(model8);
//    dictionary.push_back(model9);
//    dictionary.push_back(model10);
//    
//    std::cout<< "start recording" << std::endl;
//    
//    RawRecordData * data = record();
//    
//    std::cout<< "end recording" << std::endl;
//    
//    if (data) {
//        std::vector<MFCC_Feature> mfcc;
//        
//        handler.getMFCCFeature(mfcc, data);
//        
//        std::vector<std::vector<std::vector<std::vector<float> > > > containers(dictionary.size());
//        
//        for (int i = 0; i < dictionary.size(); i++) {
//            containers[i].resize(dictionary[i]->getNumberOfStates());
//        }
//        
//        std::vector<int> configure;
//
//        configure.push_back(10);
//        configure.push_back(0);
//        configure.push_back(1);
//        configure.push_back(2);
//        configure.push_back(3);
//        configure.push_back(4);
//        configure.push_back(5);
//        configure.push_back(6);
//        configure.push_back(7);
//        configure.push_back(8);
//        configure.push_back(9);
//        configure.push_back(10);
//
//        
//        delete gallery.DTW_sequence_structure_configured(dictionary, configure, mfcc, containers);
//        
//        for (int i = 0; i < dictionary.size(); i++) {
//            for (int j = 0; j < dictionary[i]->getNumberOfStates(); j++) {
//                std::cout<<i<<":"<<j<<":"<<containers[i][j].size()<<"\n";
//            }
//        }
//    }
//    
//    
//    
//    delete model0;
//    delete model1;
//    delete model2;
//    delete model3;
//    delete model4;
//    delete model5;
//    delete model6;
//    delete model7;
//    delete model8;
//    delete model9;
//    delete model10;
//    
//}

/* 
 
 ---------------------------------- recognize with isolated HMM
 
 */


//int main(int argc, const char * argv[]) {
//    
//    MFCCHandler handler;
//    
//    DTWGallery gallery;
//    
//    HMMModel * model0 = new HMMModelGMM();
//    model0->import("/Users/apple/Desktop/mymodel/zero.txt");
//    
//    HMMModel * model1 = new HMMModelGMM();
//    model1->import("/Users/apple/Desktop/mymodel/one.txt");
//    
//    HMMModel * model2 = new HMMModelGMM();
//    model2->import("/Users/apple/Desktop/mymodel/two.txt");
//    
//    HMMModel * model3 = new HMMModelGMM();
//    model3->import("/Users/apple/Desktop/mymodel/three.txt");
//    
//    HMMModel * model4 = new HMMModelGMM();
//    model4->import("/Users/apple/Desktop/mymodel/four.txt");
//    
//    HMMModel * model5 = new HMMModelGMM();
//    model5->import("/Users/apple/Desktop/mymodel/five.txt");
//    
//    HMMModel * model6 = new HMMModelGMM();
//    model6->import("/Users/apple/Desktop/mymodel/six.txt");
//    
//    HMMModel * model7 = new HMMModelGMM();
//    model7->import("/Users/apple/Desktop/mymodel/seven.txt");
//    
//    HMMModel * model8 = new HMMModelGMM();
//    model8->import("/Users/apple/Desktop/mymodel/eight.txt");
//    
//    HMMModel * model9 = new HMMModelGMM();
//    model9->import("/Users/apple/Desktop/mymodel/nine.txt");
//    
//    HMMModel * model10 = new HMMModelGMM();
//    model10->import("/Users/apple/Desktop/mymodel/sil.txt");
//    
////    HMMModel * model11 = new HMMModelGMM();
////    model11->import("/Users/apple/Desktop/mymodel/oh.txt");
//    
//    std::cout<<"start record"<<std::endl;
//    
//    RawRecordData * data = record();
//    
//    std::cout<<"end record"<<std::endl;
//    
//    std::vector<std::vector<float>> result;
//    
//    std::vector<HMMModel *> dictionary;
//    dictionary.push_back(model0);
//    dictionary.push_back(model1);
//    dictionary.push_back(model2);
//    dictionary.push_back(model3);
//    dictionary.push_back(model4);
//    dictionary.push_back(model5);
//    dictionary.push_back(model6);
//    dictionary.push_back(model7);
//    dictionary.push_back(model8);
//    dictionary.push_back(model9);
//    dictionary.push_back(model10);
////    dictionary.push_back(model11);
//    
//    if (data) {
//        handler.getMFCCFeature(result, data);
//        
////        int num_frames = (int)result.size();
////        
////        std::ofstream ofs;
////        
////        ofs.open("/Users/apple/Desktop/3456.txt");
////        
////        for (int i = 0; i < num_frames-1; i++) {
////            for (int j = 0; j < MFCC_FEATURE_LENGTH-1; j++) {
////                ofs<<result[i][j]<<' ';
////            }
////            ofs<<result[i][MFCC_FEATURE_LENGTH-1]<<'\n';
////        }
////        
////        for (int j = 0; j < MFCC_FEATURE_LENGTH-1; j++) {
////            ofs<<result[num_frames-1][j]<<' ';
////        }
////        ofs<<result[num_frames-1][MFCC_FEATURE_LENGTH-1]<<'\n';
////        
////        ofs.close();
//        
////        std::cout << gallery.DTW_single_digit(dictionary, result)<<std::endl;
//        
//        gallery.DTW_sequence_structure1(dictionary, result);
//        
//    }
//    
//    delete model0;
//    delete model1;
//    delete model2;
//    delete model3;
//    delete model4;
//    delete model5;
//    delete model6;
//    delete model7;
//    delete model8;
//    delete model9;
//    delete model10;
////    delete model11;
//    
//    return 0;
//    
//}

/*
 
 ---------------------------  train isolated speech
 
 */

//int main(int argc, const char * argv[]) {
//
//
//
//    std::vector<std::vector<std::vector<float> > > samples;
//
//    MFCCHandler handler;
//
//    char path[STRING_BUFFER_SIZE];
//
//    for (int i=1; i<=5; i++) {
//        sprintf(path, "/Users/apple/Desktop/mytestaudio/oh%d.wav",i);
//        std::vector<std::vector<float>> file;
//        RawRecordData * rawdata = ReadWaveFile(path);
//        handler.getMFCCFeature(file, rawdata);
//        samples.push_back(file);
//        file.clear();
//    }
//
////    SegmentalKmeansGMM skm;
////    skm.setParameters(3,FEATURE_LENGTH,4);
////    skm.commit(samples);
////    skm.output("/Users/apple/Desktop/mymodel/oh.txt");
//    
//    BaumWelchHMM bw;
//    bw.setParameters(5, FEATURE_LENGTH, 4);
//    bw.commit(samples);
//    bw.output("/Users/apple/Desktop/mymodel/oh.txt");
//
////    HMMModel * model = new HMMModelGMM();
////
////    model->import("/Users/apple/Desktop/a.txt");
////
////    delete model;
//
//    printf("done\n");
//
//    return 0;
//}
