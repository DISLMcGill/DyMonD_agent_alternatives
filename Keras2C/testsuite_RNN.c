#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "./include/k2c_include.h"
#include "RNN.h"

//float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2);

//struct timeval GetTimeStamp();

int main(){

    double cpu_time;
    struct timespec start_time, end_time;
    time_t start, end;

	clock_gettime(CLOCK_MONOTONIC, &start_time);
    float* test1_lstm_input_input_array = (float*)malloc(sizeof(float)*784); 

	
	
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0 ;
    printf("Initial Formatting  %f s\n", time_taken);

	// read from file
	clock_gettime(CLOCK_MONOTONIC, &start_time);
    FILE* file3 = fopen("../../DyMonD_Research/RNN/RNN_test_data.csv", "r");
	//float* all_data = k2c_read_array("../../DyMonD_Research/Data/test_no_labels.csv",3600*3000);
	
	float* all_data = (float*)malloc(sizeof(float)*784*10000); 
	
    int index = 0;
    char line[1000000]; 
    while (fgets(line, sizeof(line), file3) != NULL){
	char* token = strtok(line, ",");
        int count = 0;
	while (token != NULL) {
	    char *endptr;
	    all_data[index] = strtof(token,&endptr); // / 255.0;
            index++;
            count++;
	    token = strtok(NULL, ",\n");
        }
    } 
    
    //float* test1_conv2d_3_input_input_array = k2c_read_array("../DyMonD_Research/Data/test_no_labels.csv",3600*5643);
	
	
	clock_gettime(CLOCK_MONOTONIC, &end_time); // END TIME
    time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0 ;
    printf("Read data from file  %f s\n", time_taken);

	
	// normalize
	clock_gettime(CLOCK_MONOTONIC, &start_time);
    float min_val;
    float max_val;
    for(int i = 0 ; i < 784*10000; i ++){
		
        if (all_data[i] < min_val) min_val = all_data[i];
        if(all_data[i] > max_val) max_val = all_data[i];
    }
	printf("max val is %f\n", max_val);
	printf("min val is %f\n", min_val);

	for(int i=0 ; i < 784*10000; i ++){
		all_data[i] = (all_data[i]-min_val)/(max_val - min_val);
	}
	
	clock_gettime(CLOCK_MONOTONIC, &end_time); // END TIME
	time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0 ;
    printf("Normalize data  %f s\n", time_taken);

	
	// make input and output tensors
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	
	k2c_tensor test1_lstm_input_input= {&test1_lstm_input_input_array[0],2,784,{28,  28, 1,  1,  1}};
	float c_dense_test1_array[10] = {0}; 
	k2c_tensor c_dense_test1 = {&c_dense_test1_array[0],1,10,{10,1,1,1,1}};
	
	clock_gettime(CLOCK_MONOTONIC, &end_time); // END TIME
	time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0 ;
    printf("Input and output tensors  %f s\n", time_taken);

	clock_gettime(CLOCK_MONOTONIC, &start_time);
    RNN_initialize();
    
	clock_gettime(CLOCK_MONOTONIC, &end_time); // END TIME
	time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0 ;
    printf("Time to initialize/ create a session  %f s\n", time_taken);

    double total_sessions = 0; 
    double total_formatting = 0;

    FILE *file = fopen("RNN_predictions.txt", "w");
	
	
    for(int i = 0 ; i < 10000; i ++){
	
        clock_gettime(CLOCK_MONOTONIC, &start_time);

        for(int j = 0 ; j < 784 ; j ++){
            test1_lstm_input_input_array[j] = all_data[784*i + j];
        } 
		
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0 ;
        total_formatting += time_taken;

        clock_gettime(CLOCK_MONOTONIC, &start_time);
		RNN(&test1_lstm_input_input,&c_dense_test1);
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0 ;
        total_sessions += time_taken;

	
		float max_val = 0;
		int max_index = 0;
		
		//for (int k=0; k < 9; k ++){
			
			/*if(c_dense_4_test1_array[k] > max_val){
				max_val = c_dense_4_test1_array[k];
				max_index = k;
			} */
			
			// test2 *???
		int k = 0;
        fprintf(file, "%f %f %f %f %f %f %f %f %f\n", c_dense_test1_array[k],c_dense_test1_array[k+1],c_dense_test1_array[k+2],c_dense_test1_array[k+3], c_dense_test1_array[k+4], c_dense_test1_array[k+5], c_dense_test1_array[k+6], c_dense_test1_array[k+7], c_dense_test1_array[k+8]); 
		//}	
	
		//fprintf(file, "%d\n", max_index);
		

    } 
	//clock_gettime(CLOCK_MONOTONIC, &end_time); 
	//time_taken = (double)(end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0 ;
    printf("Time to run all predictions  %f s\n", total_sessions);
    printf("Avg time for 1 prediction  %f s\n", total_sessions/10000.0);

    printf("Time for all data formatting %f s\n", total_formatting);
    
    
    //errors[0] = maxabs(&keras_dense_4_test2,&c_dense_4_test2);
    /*for (int i=0; i < 31*9; i +=9){ // test2 *???
        fprintf(file, "%f %f %f %f %f %f %f %f %f\n ", c_dense_4_test1_array[i],c_dense_4_test1_array[i+1],c_dense_4_test1_array[i+2],c_dense_4_test1_array[i+3], c_dense_4_test1_array[i+4], c_dense_4_test1_array[i+5], c_dense_4_test1_array[i+6], c_dense_4_test1_array[i+7], c_dense_4_test1_array[i+8]); 
	} */
   

	fclose(file);
	RNN_terminate();
    //RNN_terminate(conv2d_3_output_array,conv2d_3_padded_input_array,conv2d_3_kernel_array,conv2d_3_bias_array,batch_normalization_3_output_array,batch_normalization_3_mean_array,batch_normalization_3_stdev_array,batch_normalization_3_gamma_array,batch_normalization_3_beta_array,conv2d_4_output_array,conv2d_4_padded_input_array,conv2d_4_kernel_array,conv2d_4_bias_array,batch_normalization_4_output_array,batch_normalization_4_mean_array,batch_normalization_4_stdev_array,batch_normalization_4_gamma_array,batch_normalization_4_beta_array,time_distributed_3_output_array,max_pooling1d_2_output_array,max_pooling1d_2_timeslice_input_array,max_pooling1d_2_timeslice_output_array,time_distributed_4_output_array,flatten_2_output_array,flatten_2_timeslice_input_array,flatten_2_timeslice_output_array,backward_lstm_2_output_array,backward_lstm_2_kernel_array,backward_lstm_2_recurrent_kernel_array,backward_lstm_2_bias_array,forward_lstm_2_output_array,forward_lstm_2_kernel_array,forward_lstm_2_recurrent_kernel_array,forward_lstm_2_bias_array,bidirectional_2_output_array,dense_3_output_array,dense_3_kernel_array,dense_3_bias_array,dense_4_kernel_array,dense_4_bias_array); 
	free(all_data);
	
}	
