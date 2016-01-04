//
//  RecorderHelper.hpp
//  speech3
//
//  Created by Dajian on 15/11/23.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#ifndef RecorderHelper_hpp
#define RecorderHelper_hpp

#include <stdio.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <string>
#include "portaudio.h"
#include "Constants.h"

struct WavFileHead
{
    //Resource Interchange File Flag (0-3) "RIFF"
    char RIFF[4];
    //File Length ( not include 8 bytes from the beginning ) (4-7)
    int FileLength;
    //WAVE File Flag (8-15) "WAVEfmt "
    char WAVEfmt_[8];
    //Transitory Byte ( normally it is 10H 00H 00H 00H ) (16-19)
    unsigned int noUse;
    //Format Category ( normally it is 1 means PCM-u Law ) (20-21)
    short FormatCategory;
    //NChannels (22-23)
    short NChannels;
    //Sample Rate (24-27)
    int SampleRate;
    //l=NChannels*SampleRate*NBitsPersample/8 (28-31)
    int SampleBytes;
    //i=NChannels*NBitsPersample/8 (32-33)
    short BytesPerSample;
    //NBitsPersample (34-35)
    short NBitsPersample;
    //Data Flag (36-39) "data"
    char data[4];
    //Raw Data File Length (40-43)
    int RawDataFileLength;
}__attribute((packed));

RawRecordData * ReadWaveFile(const char * path);

bool WaveRewind(FILE *wav_file, WavFileHead *wavFileHead);

// used in pa_callbackfunc
bool isSilent();

// reset global variables, used in record
void reset();

// commit a record, return NULL if failed
RawRecordData * record();

// energy function, used in pa_callbackfunc
float Energy(short *chunk);

// call back function of PortAudio, used in record
int pa_callbackfunc(const void *inputBuffer, void *outputBuffer,
                    unsigned long framesPerBuffer,
                    const PaStreamCallbackTimeInfo* timeInfo,
                    PaStreamCallbackFlags statusFlags,
                    void *userData);

// read a wav file


#endif /* RecorderHelper_hpp */
