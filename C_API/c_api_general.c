#include <stdlib.h>
#include <stdio.h>
#include "../../libtensorflow/include/tensorflow/c/c_api.h"
#include <time.h>

// DyMonD:
// Batch size 32: BATCH_SIZE 32 ; NUM_TESTS 5632 ; NUM_BATCHES 176
// Batch size 31: BATCH_SIZE 31 ; NUM_TESTS 5642 ; NUM_BATCHES 182
// Batch size 5643: BATCH_SIZE 5643 ; NUM_TESTS 5643 ; NUM_BATCHES 1
// Batch size 1: BATCH_SIZE 1 ; NUM_TESTS 5643 ; NUM_BATCHES 5643
// DIMS: 100, 36, 1
// OUT_DIM1: 10

// Resnet:
// Batch size 10000: BATCH_SIZE 10000 ; NUM_TESTS 10000 ; NUM_BATCHES 1
// Batch size 1: BATCH_SIZE 1 ; NUM_TESTS 10000 ; NUM_BATCHES 10000
// Batch size 32: BATCH_SIZE 32 ; NUM_TESTS 10000 ; NUM_BATCHES 312
// DIMS: 32, 32, 3
// OUT_DIM1: 10

// RNN:
// Batch size 1: BATCH_SIZE 1 ; NUM_TESTS 10000 ; NUM_BATCHES 10000
// Batch size 10000: BATCH_SIZE 10000 ; NUM_TESTS 10000 ; NUM_BATCHES 1

// DIMS: 28, 28
// OUT_DIM1: 10

#define _POSIX_C_SOURCE 200809L

#define NUM_TESTS 10000
#define DIM1 32
#define DIM2 32
#define DIM3 3
#define OUT_DIM1 10
// DyMonD: "../Python/10classesmodelpb/model_0/" (currently have "../Models/model_0/" )
// Resnet: "/Users/izzychampion/DyMonD_Research/ResNet/resnetpb_/"
        //"/Users/izzychampion/DyMonD_Research/ResNet/Models/model_/"
// RNN: "/Users/izzychampion/DyMonD_Research/RNN/RNNpb"
#define MODEL_PATH  "/Users/izzychampion/DyMonD_Research/ResNet/resnetpb_/"
#define SERVE_TAG "serve"
#define NUM_INPUTS 1 // number of inputs to model (not batch size)
#define NUM_OUTPUTS 1
// DyMond, Resnet: 4 // RNN: 3
#define NUM_DIMS 4 
#define BATCH_SIZE 10000
#define NUM_BATCHES 1
#define N_DATA sizeof(float)*DIM1*DIM2*DIM3*BATCH_SIZE // number of bytes in one input
// ResNet: "/Users/izzychampion/DyMonD_Research/ResNet/ResnetTestData.csv"
// DyMonD:  "../Data/test_no_labels.csv"
// RNN: "/Users/izzychampion/DyMonD_Research/RNN/RNN_test_data.csv"
#define PathToTestData "/Users/izzychampion/DyMonD_Research/ResNet/ResnetTestData.csv"
// possibly >1 inputs
// ResNet: "serving_default_resnet50v2_input"
// DyMonD: "serving_default_conv2d_input"
// RNN: "serving_default_lstm_input"
#define INPUT1_NAME "serving_default_resnet50v2_input"
// possibly >1 outputs 
#define OUTPUT1_NAME "StatefulPartitionedCall"


void NoOpDeallocator(void* data, size_t a, void* b) {}

int main()
{
    
    double cpu_time;
    struct timespec start_time, end_time;
    time_t start, end;

    /*** INITIAL FORMATTING ***/
    clock_gettime(CLOCK_MONOTONIC, &start_time); // START TIME
	
    TF_Graph* Graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();
    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL; 
    const char* saved_model_dir = MODEL_PATH; 
    const char* tags = SERVE_TAG; // default model serving tag
    int ntags = 1; // ?
    
    clock_gettime(CLOCK_MONOTONIC, &end_time); // END TIME
    double time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0 ;
    printf("Initial processing::  %f s\n", time_taken);
    
    /*** CREATE A SESSION ***/
    clock_gettime(CLOCK_MONOTONIC, &start_time); // START TIME
    
    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    
    clock_gettime(CLOCK_MONOTONIC, &end_time); // END TIME
    time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0 ;
    printf("Create a session::  %f s\n", time_taken); 
    
    //*** CHECK IF ERROR CREATING SESSION ***//
    if(TF_GetCode(Status) == TF_OK)
        printf("TF_LoadSessionFromSavedModel OK\n"); 
    else
	    printf("%s",TF_Message(Status));
    
    /*** GET INPUT & OUTPUT TENSORS ***/
    clock_gettime(CLOCK_MONOTONIC, &start_time); // START TIME
    
    TF_Output* Input = malloc(sizeof(TF_Output) * NUM_INPUTS);
    // MANUALLY CHANGE DEPENING ON NUM INPUTS 
    TF_Output t0 = {TF_GraphOperationByName(Graph, INPUT1_NAME), 0};
    Input[0] = t0;
    
    TF_Output* Output = malloc(sizeof(TF_Output) * NUM_OUTPUTS);
    // MANUALLY CHANGE DEPENDING ON NUM OUTPUTS 
    TF_Output t2 = {TF_GraphOperationByName(Graph, OUTPUT1_NAME), 0}; // 0 ?
    Output[0] = t2;

    // To hold input & output data
    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NUM_INPUTS);
    TF_Tensor** OutputValues = malloc(sizeof(TF_Tensor*)*NUM_OUTPUTS);

    float* Data_array = (float*)malloc(BATCH_SIZE*(DIM1*DIM2*DIM3) * sizeof(float)); // to hold input data
    
    // MANUALLY CHANGE FOR >3 DIM INPUT
    int64_t dims[] = {BATCH_SIZE,DIM1,DIM2,DIM3}; 
    
    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, NUM_DIMS, Data_array, N_DATA, &NoOpDeallocator, 0);
    InputValues[0] = int_tensor;

    clock_gettime(CLOCK_MONOTONIC, &end_time); // END TIME
    time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0 ;
    printf("Make input/output tensors: %f s\n", time_taken);


    
    
    //*** CHECK FOR ERROR w INPUT/OUTPUT TENSORS ***//

    if(t0.oper == NULL)
        printf("Error: Failed TF_GraphOperationByName %s\n",INPUT1_NAME);
    else    
        printf("TF_GraphOperationByName %s is OK\n", INPUT1_NAME); 
    if(t2.oper == NULL)
        printf("Error: Failed TF_GraphOperationByName %s\n",OUTPUT1_NAME);
    else
        printf("TF_GraphOperationByName %s is OK\n", OUTPUT1_NAME); 
    
    
    //*** READ DATA FROM A CSV FILE ***//
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    float* array;
    // MANUALLY CHANGE FOR >3 DIM INPUT
    array = (float*)malloc(NUM_TESTS*DIM1*DIM2*DIM3 * sizeof(float)); // to hold data from file
   
    FILE* file = fopen(PathToTestData, "r");
    if (array == NULL) {
        printf("Memory allocation failed.\n");
        fclose(file);
        return 1;
    }
   
    int index = 0;
    char line[1000000]; // change
    int counter = 0;

    while (fgets(line, sizeof(line), file) != NULL && counter < NUM_TESTS){
        counter += 1;
	    char* token = strtok(line, ",");
        int count = 0;
	    while (token != NULL) {
	        char *endptr;
	        array[index] = strtof(token,&endptr); // / 255.0;
            index++;
            count++;
	        token = strtok(NULL, ",\n");
        }
    } 
    fclose(file);
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0 ;
    printf("Read data from file:  %f s\n", time_taken);
    

    //*** MIN MAX NORMALIZATION ***//
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    float min_value, max_value;
    // MANUALLY CHANGE FOR >3 DIM INPUT
    for(int i = 0 ; i < NUM_TESTS*DIM1*DIM2*DIM3; i ++){
        if(array[i] > max_value) max_value = array[i];
        if(array[i] < min_value) min_value = array[i];
    }
    
    // normalize
    for(int i=0 ; i < NUM_TESTS*DIM1*DIM2*DIM3; i ++){
        array[i] = (array[i] - min_value)/(max_value - min_value);
    } 
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + ((double) (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0 );
    printf("Min Max Normalization: %f (seconds) \n", time_taken);

    FILE* file2 = fopen("ten_classes_0_results.csv", "w");
    void* buff;
    float* offsets;
    double total_sessions = 0; // sum
    double total_formatting = 0;
    

    for(int i = 0 ; i < NUM_BATCHES ; i ++){
        int k = 0;
    
        //*** SHAPING INPUT DATA  ***//
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        // MANUALLY CHANGE FOR >3 DIM INPUT
        for (int s = 0; s < BATCH_SIZE*(DIM1*DIM2*DIM3); s ++){
            Data_array[s] = array[BATCH_SIZE*(DIM1*DIM2*DIM3)*i + s];
        }
    
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0 ;
        total_formatting += time_taken;
        
        //*** RUN SESSION ***//
        
        clock_gettime(CLOCK_MONOTONIC, &start_time);
 
        TF_SessionRun(Session, NULL, Input, InputValues, NUM_INPUTS, Output, OutputValues, NUM_OUTPUTS, NULL, 0,NULL , Status);
     
        clock_gettime(CLOCK_MONOTONIC, &end_time);
       
        time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0 ;
        total_sessions += time_taken;
        //printf("Time taken %f\n", time_taken);

        //*** CHECK IF SESSION OK ***//
        if(TF_GetCode(Status) != TF_OK)
            printf("%s",TF_Message(Status));
            
        
        //*** WRITE OUTPUT TO FILE ***//
        buff = TF_TensorData(OutputValues[0]);
        offsets = buff;
        /*for(int j =0; j < BATCH_SIZE*OUT_DIM1; j ++){
            for(int l = 0; l < OUT_DIM1; l ++){
                fprintf(file2,"%f", offsets[j+l]);
            }fprintf(file,"\n");
        } */
        
        for(int j = 0 ; j < BATCH_SIZE*OUT_DIM1; j += 10){
            fprintf(file2,"%f %f %f %f %f %f %f %f %f %f\n",offsets[j], offsets[j+1], offsets[j+2],offsets[j+3],offsets[j+4],offsets[j+5],offsets[j+6]
	      ,offsets[j+7],offsets[j+8], offsets[j+9] );
          
        }
    }
    printf("Time for all sessions: %f s\n", total_sessions);
    printf("Average time per prediction: %f s\n", total_sessions/(NUM_TESTS));
    printf("Average time per session: %f s\n",total_sessions/NUM_BATCHES);
    printf("Time for formatting %f\n", total_formatting);
    
    //*** CLOSE SESSION ***//
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);
    
    fclose(file2); 
    
    free(array);
    return 0;
    
}
