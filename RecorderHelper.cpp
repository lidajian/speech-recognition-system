//
//  RecorderHelper.cpp
//  speech3
//
//  Created by Dajian on 15/11/23.
//  Copyright © 2015年 Dajian. All rights reserved.
//

#include "RecorderHelper.hpp"

int count_chunk = 0;
float background_energy = 0;
float smooth_energy = 0;
int SilenceMode = true;
int count_speech = 0;
int count_silence = 0;

bool WaveRewind(FILE *wav_file, WavFileHead *wavFileHead){
    char riff[8],wavefmt[8];
    short i;
    rewind(wav_file);
    fread(wavFileHead,sizeof(struct WavFileHead),1,wav_file);
    
    for ( i=0;i<8;i++ )
    {
        riff[i]=wavFileHead->RIFF[i];
        wavefmt[i]=wavFileHead->WAVEfmt_[i];
    }
    riff[4]='\0';
    wavefmt[7]='\0';
    if ( strcmp(riff,"RIFF")==0 && strcmp(wavefmt,"WAVEfmt")==0 )
        return	true;  // It is WAV file.
    else
    {
        rewind(wav_file);
        return(false);
    }
}

RawRecordData * ReadWaveFile(const char * path){
    
    FILE * wavFp = fopen(path, "rb");
    
    if (wavFp) {
        
        WavFileHead wavhead;
        
        if (WaveRewind(wavFp, &wavhead)) {
            
            RawRecordData * data = new RawRecordData();
            
            data->samplerate = wavhead.SampleRate;
            
            data->start = 0;
            
            int temp = wavhead.RawDataFileLength / sizeof(short);
            
            data->end = temp - 1;
            
            data->data.resize(temp);
            
            fread(data->data.data(), sizeof(short), temp, wavFp);
            
            fclose(wavFp);
            
            return data;
        }
        
    }
    
    return NULL;
}

// double threshold algorithm
bool isSilent(){
    
    float temp = smooth_energy - background_energy;
    
    if (SilenceMode) {
        if (temp > ENTRY_THRESHOLD) {
            return false;
        } else{
            return true;
        }
    } else{
        if (temp < EXIT_THRESHOLD) {
            return true;
        } else{
            return false;
        }
    }
    
}

void reset(){
    count_chunk = 0;
    SilenceMode = true;
    count_speech = 0;
    count_silence = 0;
    background_energy = 0;
    smooth_energy = 0;
}

// calculate the energy of every chunk
float Energy(short * chunk)
{
    float sum = 0;
    for(int i = 0; i < FRAMES_PER_BUFFER; i++)
    {
        float temp = (float)(*chunk);
        chunk++;
        sum = temp * temp;
    }
    return 10 * logf(sum + 1);
}

// call back function
int pa_callbackfunc(const void *inputBuffer, void *outputBuffer,
                    unsigned long framesPerBuffer,
                    const PaStreamCallbackTimeInfo* timeInfo,
                    PaStreamCallbackFlags statusFlags,
                    void *userData){
    
    RawRecordData * data = (RawRecordData *)userData;
    short * chunk = (short *)inputBuffer;
    
    if (count_chunk < DROP_CHUNKS) {
        // initialize background energy
        if (count_chunk >= 5) {
            background_energy += Energy(chunk) / 5;
            smooth_energy = background_energy;
        }

    } else{
        
        short * ptr_chunk = chunk;
        
        // put in cache
        for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
            data->data.push_back(*ptr_chunk);
            ptr_chunk++;
        }
        
        // smooth the energy
        smooth_energy = ((smooth_energy * FORGET_FACTOR) + Energy(chunk)) / (1 + FORGET_FACTOR);
        
        if (SilenceMode) {
            
            if (isSilent()) {
                count_speech = 0;
                count_silence++;
                background_energy += (smooth_energy - background_energy) * BACKGROUND_ADJUSTMENT;
                if (count_silence > MAX_SILENCE_LENGTH) {
                    return paComplete;
                }
            } else{
                count_silence = 0;
                count_speech++;
                if (count_speech > VALID_SPEECH_CHUNKS) {
                    count_speech = 0;
                    SilenceMode = false;
                    int temp = count_chunk - DROP_CHUNKS - VALID_SPEECH_CHUNKS - SILENCE_CHUNK_RETAINED + 1;
                    data->start = (temp<0 ? 0:temp) * FRAMES_PER_BUFFER  - 1;
                }
            }
            
        }else{
            
            if (isSilent()) {
                count_silence++;
                
                if (count_speech <= VALID_SPEECH_CHUNKS) {
                    count_silence += count_speech;
                }
                count_speech = 0;
                if (count_silence > SILENCE_CHUNK_RETAINED) {
                    data->end = (int)data->data.size() - 1;
                    return paComplete;
                }
            }else{
                count_speech++;
                if (count_speech > VALID_SPEECH_CHUNKS) {
                    count_silence = 0;
                }
            }
            
        }
        
    }
    
    count_chunk++;
    return paContinue;
    
}

// use PortAudio to record
RawRecordData * record(){
    
    reset();
    
    RawRecordData * data = new RawRecordData();
    
    PaStream * stream;
    
    PaError ret;
    
    if((ret = Pa_Initialize()) == paNoError){
        
        PaStreamParameters parameters;
        
        if((parameters.device = Pa_GetDefaultInputDevice()) != paNoDevice){
            
            parameters.hostApiSpecificStreamInfo = NULL;
            
            parameters.channelCount = 1;
            
            parameters.sampleFormat = paInt16;
            
            parameters.suggestedLatency = Pa_GetDeviceInfo(parameters.device) -> defaultLowInputLatency;
            
            if ((ret = Pa_OpenStream(&stream, &parameters, NULL, SAMPLE_RATE, FRAMES_PER_BUFFER, NULL, pa_callbackfunc, data)) == paNoError){
                
                if ((ret = Pa_StartStream(stream)) == paNoError) {
                    
                    while ((ret = Pa_IsStreamActive(stream)) == 1);
                    
                    if ((ret = Pa_StopStream(stream) ) != paNoError) {
                        std::cout << "error when closing the stream!" << std::endl;
                    }
                    
                    // invalid record, ignore it
                    if (data->end - data->start < 2 * (SILENCE_CHUNK_RETAINED + VALID_SPEECH_CHUNKS) * FRAMES_PER_BUFFER) {
                        delete data;
                        data = NULL;
                    }
                    
                    return data;

                }
                
            }
        }
        
    }
    
    delete data;
    return NULL;
    
}

