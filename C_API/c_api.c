#include <stdlib.h>
#include <stdio.h>
#include "../libtensorflow/include/tensorflow/c/c_api.h"
#include <time.h>

#define _POSIX_C_SOURCE 200809L

// 5643 tests

void NoOpDeallocator(void* data, size_t a, void* b) {}

int main()
{
	//********* Read model
    TF_Graph* Graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();

    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;

    const char* saved_model_dir = "../Models/model_0/"; // Path of the model
    //const char* saved_model_dir = "SaveLoadpbModel/Models/Models/model_0/";
    
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;
    
    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    if(TF_GetCode(Status) == TF_OK)
    {
        printf("TF_LoadSessionFromSavedModel OK\n");
    }
    else
    {
    
	printf("%s",TF_Message(Status));
    }
    
    //****** Get input tensor
    int NumInputs = 1;
    
    TF_Output* Input = malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_conv2d_input"), 0};
    
    if(t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_conv2d_input\n");
    else
	    printf("TF_GraphOperationByName serving_default_conv2d_input is OK\n");
    

    Input[0] = t0; 
    
    //********* Get Output tensor
    int NumOutputs = 1;
    
    TF_Output* Output = malloc(sizeof(TF_Output) * NumOutputs);
    
    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    
    if(t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else
	printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

    Output[0] = t2;

    //********* Allocate data for inputs & outputs
    
    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    TF_Tensor** OutputValues = malloc(sizeof(TF_Tensor*)*NumOutputs);

    int ndims = 4;
    int64_t dims[] = {5643,100,6,6};
    
    // timing the formatting
    double cpu_time;
    struct timespec start_time, end_time;
    
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    FILE* file = fopen("../Data/test_no_labels.csv", "r");
    
    float* array;
    array = (float*)malloc(5643*3600 * sizeof(float));
    
    if (array == NULL) {
        printf("Memory allocation failed.\n");
        fclose(file);
        return 1;
    }

    int index = 0;
    
    char line[1000000]; 
    while (fgets(line, sizeof(line), file) != NULL){
	char* token = strtok(line, ",");
        int count = 0;
	while (token != NULL) {
	    char *endptr;
	    array[index] = strtof(token,&endptr);
            // array[index] = atoi(token);
            index++;
            count++;
	    token = strtok(NULL, ",\n");
        }
    }
     
    fclose(file);
    clock_gettime(CLOCK_MONOTONIC, &end_time);	
    double time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0 ;
    printf("Time to format: %f s \n", time_taken);

    clock_gettime(CLOCK_MONOTONIC, &start_time);	
    float *NormalizedT;
    NormalizedT = (float*)malloc(5643*3600 * 32);
    
    // normalizing data
    for(int i = 0; i < 5643 ; i++){ 
	for(int j = 0; j < 3600 ; j++){
                NormalizedT[3600 * i + j] = (array[3600 * i + j] - 0) / (255.0 - 0);
            }
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0 ;
    printf("Time to normalize: %f s \n ", time_taken);

    int ndata = sizeof(float)*3600*5643; // number of bytes in input
    
    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, NormalizedT, ndata, &NoOpDeallocator, 0);
    
    if (int_tensor != NULL) 
    {
        printf("TF_NewTensor is OK\n");
    }
    else
	printf("ERROR: Failed TF_NewTensor\n");
    
    
    InputValues[0] = int_tensor;
    
    // time
    
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0,NULL , Status);
    // divide time by num records
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0 ;
    printf("\nTime taken for session run: %f s \n\n", time_taken );    
    printf("Time taken per test: %f\n", time_taken/5643);
    
    const char* error_message = TF_Message(Status);
    
    if(TF_GetCode(Status) == TF_OK)
    {
        printf("Session is OK\n");
    }
    else
    {
        printf("%s",TF_Message(Status));
    }
    // Free memory
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);

    void* buff = TF_TensorData(OutputValues[0]);
    
    float* offsets = buff;
     
    FILE* file2 = fopen("ten_classes_0_results.csv", "w");
    int i = 0;
    
    // write predictions to ten_classes_0_results.csv

    while(i < 5643*10){
	fprintf(file2,"%f %f %f %f %f %f %f %f %f %f\n",offsets[i], offsets[i+1], offsets[i+2],offsets[i+3],offsets[i+4],offsets[i+5],offsets[i+6]
	      ,offsets[i+7],offsets[i+8], offsets[i+9] );
	i++; i++;i++; i++;i++; i++;i++; i++;i++;i++;
    }
    
    fclose(file2);
    
    free(array);
    return 0;
}
