#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>

// DyMonD
// Batch size 32: BATCH_SIZE 32 ; NUM_TESTS 5632 ; NUM_BATCHES 176
// Batch size 31: BATCH_SIZE 31 ; NUM_TESTS 5642 ; NUM_BATCHES 182
// Batch size 5643: BATCH_SIZE 5643 ; NUM_TESTS 5643 ; NUM_BATCHES 1
// Batch size 1: BATCH_SIZE 1 ; NUM_TESTS 5643 ; NUM_BATCHES 5643
// DIMS: 100, 36, 1
// OUT_DIM1: 10

// ResNet
// Batch size 10000: BATCH_SIZE 10000 ; NUM_TESTS 10000 ; NUM_BATCHES 1
// Batch size 1: BATCH_SIZE 1 ; NUM_TESTS 10000 ; NUM_BATCHES 10000
// Batch size 32: BATCH_SIZE 32 ; NUM_TESTS 10000 ; NUM_BATCHES 312
// DIMS: 32, 32, 3
// OUT_DIM1: 10

// RNN
// Batch size 1: BATCH_SIZE 1 ; NUM_TESTS 10000 ; NUM_BATCHES 10000
// Batch size 10000: BATCH_SIZE 10000 ; NUM_TESTS 10000 ; NUM_BATCHES 1

// DIMS: 28, 28
// OUT_DIM1: 10
// #define isRNN true

#define NUM_TESTS 5643
#define DIM1 100
#define DIM2 36
#define DIM3 1
#define OUT_DIM1 10
// DyMonD: "ten_classes_0/" and "/Users/izzychampion/tenclassesmodelpb0"
// ResNet: "/Users/izzychampion/DyMonD_Research/ResNet/resnetpb_/" or "/Users/izzychampion/resnetpb1_/"
// RNN: "/Users/izzychampion/DyMonD_Research/RNN/RNNpb"
#define MODEL_PATH "/Users/izzychampion/tenclassesmodelpb0"
#define SERVE_TAG "serve"
#define NUM_INPUTS 1 // number of inputs to model (not batch size)
#define NUM_OUTPUTS 1
// DyMond, Resnet: 4 // RNN: 3
#define NUM_DIMS 4
#define BATCH_SIZE 1
#define NUM_BATCHES 5643
#define N_DATA sizeof(float)*DIM1*DIM2*DIM3*BATCH_SIZE // number of bytes in one input
// DyMonD: "revamped.csv"
// ResNet: "/Users/izzychampion/DyMonD_Research/ResNet/ResnetTestData.csv"
// RNN: "/Users/izzychampion/DyMonD_Research/RNN/RNN_test_data.csv
#define PathToTestData "revamped.csv"

// possibly >1 inputs
// DyMonD: "serving_default_conv2d_input"
// ResNet: "serving_default_resnet50v2_input"
// RNN: "serving_default_lstm_input"
#define INPUT1_NAME "serving_default_conv2d_input"
// possibly >1 outputs 
#define OUTPUT1_NAME "StatefulPartitionedCall"

using namespace tensorflow;
using namespace std;

int main(){

    //*** INITIAL PROCESSING ***//
    auto init_processing = std::chrono::steady_clock::now();	
	
    const std::string export_dir = MODEL_PATH;	
	tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
	tensorflow::SavedModelBundle model_bundle;

    //f_session.reset(NewSession(options));
    model_bundle.session.reset(NewSession(session_options));

    auto init_processing_end = std::chrono::steady_clock::now();	
    auto init_proc = std::chrono::duration_cast<std::chrono::nanoseconds>(init_processing_end - init_processing).count();
    std::cout << "Initial processing: " << init_proc << "nanoseconds" << std::endl;
    //auto fDuration = std::chrono::duration_cast<std::chrono::duration<float>>(init_processing_end - init_processing);
    //std::cout << "Initial processing(2)  = " << fDuration.count()  << std::endl;

    //*** CREATE A SESSION ***//
   	auto session_start = std::chrono::steady_clock::now();	

	tensorflow::Status status = tensorflow::LoadSavedModel(
        	session_options, run_options, export_dir, {tensorflow::kSavedModelTagServe},
        	&model_bundle);

	auto session_end = std::chrono::steady_clock::now();	
    auto cr_session = std::chrono::duration_cast<std::chrono::nanoseconds>(session_end - session_start).count();
    std::cout << "Create a session: " << cr_session << "nanoseconds" << std::endl;
    std::cout << "Create a session: " << cr_session/1000000000 << "seconds" << std::endl;
    //fDuration = std::chrono::duration_cast<std::chrono::duration<float>>(session_end - session_start);
    //std::cout << "Create a session(2)  = " << fDuration.count()  << std::endl;

    //*** CHECK IF ERROR CREATING SESSION ***//

    if (!status.ok()) {
    		std::cout << status.ToString() << "\n";
    		return 1;
  	} else {
    	std::cout << "Load saved model successfully" << std::endl;
  	}
	
    //*** GET INPUT & OUTPUT TENSORS ***//
    auto make_input = std::chrono::steady_clock::now();	

    /////////////////////////////////
    /// *** CHANGED TO DIM 3 *** ///
    ///////////////////////////////
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,tensorflow::TensorShape({BATCH_SIZE,DIM1,DIM2,DIM3}) );
	//tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,tensorflow::TensorShape({BATCH_SIZE,DIM1,DIM2}) );
	
    auto input_tensor_mapped  = input_tensor.tensor<float, NUM_DIMS>(); // to hold input data
    std::vector<tensorflow::Tensor> outputs;

    auto make_input_end = std::chrono::steady_clock::now();	
    auto make_input_tensor = std::chrono::duration_cast<std::chrono::nanoseconds>(make_input_end - make_input).count();
    std::cout << "Make input/output tensors: " << make_input_tensor << "nanoseconds" << std::endl;
    std::cout << "Make input/output tensors: " << make_input_tensor/1000000000 << "seconds" << std::endl;
    
    //fDuration = std::chrono::duration_cast<std::chrono::duration<float>>(make_input_end - make_input);
    //std::cout << "make input/output tensors(2)  = " << fDuration.count()  << std::endl;

	//*** READ DATA FROM A CSV FILE ***//
    auto file_read = std::chrono::steady_clock::now();	
	
    std::ifstream file; 
    file.open(PathToTestData);

	if(!file.is_open()){
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    int i = 0;
    float* array = static_cast<float*>(malloc(NUM_TESTS*DIM1*DIM2*DIM3 * sizeof(float)));
    int h = 0;

    std::string line;
    int index = 0;
    int counter = 0;
    while(std::getline(file,line) && counter < NUM_TESTS){
        std::istringstream aLine(line);
        std::string token;
        while(std::getline(aLine, token, ',')){
            array[index] = std::stof(token); index ++;
        }
        counter ++;

    }

    file.close();
	
    auto file_read_end = std::chrono::steady_clock::now();	
    auto read_from_file = std::chrono::duration_cast<std::chrono::milliseconds>(file_read_end - file_read).count();
    std::cout << "Read data from file: " << read_from_file << "milliseconds" << std::endl;		
    std::cout << "Read data from file: " << read_from_file/1000000000 << "seconds" << std::endl;

    //*** MIN MAX NORMALIZATION ***//
    auto min_max_start = std::chrono::steady_clock::now();	
    
    float min_val;
    float max_val;
    for (int i = 0 ; i < NUM_TESTS*DIM1*DIM2*DIM3; i ++) {
        if(array[i] > max_val) max_val = array[i];
        if(array[i] < min_val) min_val = array[i];
    }
    // normalize
    for(int i = 0 ; i < NUM_TESTS*DIM1*DIM2*DIM3; i ++){
        array[i] = (array[i]-min_val)/(max_val - min_val);
    }
    

    auto min_max_end = std::chrono::steady_clock::now();
    auto normalizing = std::chrono::duration_cast<std::chrono::nanoseconds>(min_max_end - min_max_start).count();
    std::cout << "Min Max Normalization: " << normalizing << "nanoseconds" << std::endl;
    std::cout << "Min Max Normalization: " << normalizing/1000000000 << "seconds" << std::endl;
    //fDuration = std::chrono::duration_cast<std::chrono::duration<float>>(min_max_end - min_max_start);
    //std::cout << "Min Max Normalization(2)  = " << fDuration.count()  << std::endl;

	
	std::ofstream output_file("output_predictions.txt", std::ios::app);
    long formatting_times = 0;
    long formatting_times2 = 0;

	long session_run_times = 0;
    long session_run_2= 0;
    int session_total = 0;

    for(int i=0; i < 1 ; i++){

        //*** SHAPING INPUT DATA ***//
		auto format_data = std::chrono::steady_clock::now();
    	h = 0;
    	for(int j = 0; j < BATCH_SIZE; j++ ){
            //printf("hello %d\n", j);
    		for (int m = 0 ; m < DIM1 ; m++){
				for(int n = 0 ; n < DIM2 ; n++ ){
                   
					for(int o = 0; o < DIM3 ; o++ ){
                       
                         
                        input_tensor_mapped(j,m,n,o) = array[j*DIM1*DIM2*DIM3 + m*DIM2*DIM3 + n*DIM3 + o];
                        
                		h++;
					} 
                    //input_tensor_mapped(j,m,n) = array[j*DIM1*DIM2+ m*DIM2 + n];
                    //h++;
				}
			}
    	}

		auto format_data_end = std::chrono::steady_clock::now();
		auto data_format_time = std::chrono::duration_cast<std::chrono::nanoseconds>(format_data_end - format_data).count();
		formatting_times += static_cast<float>(data_format_time);
        //fDuration = std::chrono::duration_cast<std::chrono::duration<float>>(format_data_end - format_data);
        //formatting_times2 += fDuration.count();

		//*** RUN SESSION ***//
		auto start_run = std::chrono::steady_clock::now();	
	
		status = model_bundle.session->Run({{"serving_default_conv2d_input:0",input_tensor}}, {"StatefulPartitionedCall:0"}, {}, &outputs);
  		
        auto end_run = std::chrono::steady_clock::now();
        auto session_run = std::chrono::duration_cast<std::chrono::nanoseconds>(end_run - start_run).count();
        session_run_times += static_cast<float>(session_run);
        //session_total += session_run;
      
        //fDuration = std::chrono::duration_cast<std::chrono::duration<float>>(end_run - start_run);
        //session_run_2 += fDuration.count();

        //*** CHECK IF SESSION OK ***//
		if (!status.ok()) {
    		std::cout << status.ToString() << "\n";
    		return 1;
        }

        //*** WRITE OUTPUT TO FILE***//
		tensorflow::Tensor& output_tensor = outputs[0];
        auto output_mapped = output_tensor.tensor<float, 2>();
		
        
        // loop to get maximum
		for( int x = 0; x < BATCH_SIZE ; x++){
			for(int  y = 0 ; y < OUT_DIM1 ; y ++ ){
				// get max output_mapped(x,y) y = 0 -> 9
                output_file << output_mapped(x,y);
				if (y < 9){
                    output_file << " ";
                }
			} output_file << "\n";
		} 

	}
    outputs.clear();
    
    free(array);

    std::cout << "Time for all sessions: " << session_run_times << " nanoseconds "<< std::endl;
    std::cout << "Time for all sessions: " << session_run_times/1000000000 << " seconds "<< std::endl;
    //std::cout << "Time for all sessions (2): " << session_run_2 << std::endl;
    //std::cout << "time2 for all sessions: " << session_total << std::endl;
    //std::cout << "time2 for all sessions: " << session_total << std::endl;
	std::cout << "Average session run time nanoseconds: " << session_run_times/(NUM_TESTS) << std::endl;
    //std::cout << "Average session run time seconds: " << session_run_times/((NUM_TESTS)*1000000000) << std::endl;
    //std::cout << "Average session run time milliseconds (2): " << session_run_2/((NUM_TESTS)*1.0) << std::endl;
    std::cout << "Total data processing time: " << formatting_times << " nanoseconds" << std::endl;
    std::cout << "Total data processing time: " << formatting_times/1000000000 << " seconds" << std::endl;
    //std::cout << "Total data processing time (2): " << formatting_times2 << std::endl;
    


	output_file.close();
    

	//*** CLOSE SESSION ***//
    tensorflow::Status close_status = model_bundle.session->Close();	


    if (!close_status.ok()) {
    		std::cerr << "Error closing session: " << close_status.ToString() << "\n";
	}

    return 0;

}
