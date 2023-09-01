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
#include <tensorflow/cc/tools/freeze_saved_model.h>
// #include <tensorflow/core/protobuf/meta_graph.pb.h>
// #include <tensorflow/core/framework/graph_debug_info.pb.h>
using namespace tensorflow;
using namespace std;

int main(){

	auto start_process = std::chrono::high_resolution_clock::now();	
	const std::string export_dir = "ten_classes_0/";	
	
	tensorflow::SessionOptions session_options;
        tensorflow::RunOptions run_options;
        // model_bundle session type because of format of saved model file
	tensorflow::SavedModelBundle model_bundle;
   	 
	tensorflow::Status status = tensorflow::LoadSavedModel(
        	session_options, run_options, export_dir, {tensorflow::kSavedModelTagServe},
        	&model_bundle);

	if (!status.ok()) {
    		std::cout << status.ToString() << "\n";
    		return 1;
  	} else {
    	std::cout << "Load saved model successfully" << std::endl;
  	}

	std::vector<tensorflow::Tensor> outputs;
	// change 5643 to number of inputs
	tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,tensorflow::TensorShape({5643,100,6,6}) );
	
	auto input_tensor_mapped  = input_tensor.tensor<float, 4>();	
	// revamped.csv: file with inputs (without labels), one input per line
	std::ifstream file; file.open("revamped.csv");
	std::string line;
	if(!file.is_open()){
                std::cerr << "Failed to open the file." << std::endl;
                return 1;
        }else {
        	std::cout << "File opened successfully" << std::endl;
        }
	
        int i = 0;
	// change 5643 to number of inputs
        while (std::getline(file, line) && i < 5643) {
        	std::istringstream iss(line);
		
		float value;
		// add each input in shape {1,100,6,6}
		for (int m = 0 ; m < 100 ; m++){
			for(int n = 0 ; n < 6 ; n ++){
				for(int o = 0; o < 6 ; o ++){
					if(iss >> value){
						// skip commas in file
						if (iss.peek() == ',') {
                       	 				iss.ignore();
						}
					// normalize
					input_tensor_mapped(i,m,n,o) = value/255.0;
					}
				}
			}
		}
		i++;
    	}
	file.close();	
	// time the session	
	auto start_run = std::chrono::high_resolution_clock::now();	
	// tensor to store output data
	status = model_bundle.session->Run({{"serving_default_conv2d_input:0",input_tensor}}, {"StatefulPartitionedCall:0"}, {}, &outputs);
  	auto end_run = std::chrono::high_resolution_clock::now();
	
	if (!status.ok()) {
    		std::cout << status.ToString() << "\n";
    		return 1;
  	}else {
    		std::cout << "Success running session: " << status.ToString() << "\n";
	}
	// file for output predictions
	std::ofstream output_file("output_predictions.txt");
	
	// format output data according to shape {1,10} -> per output, total shape is {5643,10}
	tensorflow::Tensor& output_tensor = outputs[0];
        auto output_mapped = output_tensor.tensor<float, 2>();
	for( int x = 0; x < 5643 ; x++){
		for(int  y = 0 ; y < 10 ; y ++ ){
			output_file << output_mapped(x,y);
			if (y < 9){
                                output_file << " ";
                        }
		}
		output_file << "\n";
	}	

	output_file.close();

        tensorflow::Status close_status = model_bundle.session->Close();		
	
	if (!close_status.ok()) {
    		std::cerr << "Error closing session: " << close_status.ToString() << "\n";
	}
	auto end_process = std::chrono::high_resolution_clock::now();
	
	// time calculations
	auto process_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_process - start_process).count();
	auto run_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_run - start_run).count();
	
	std::cout << "Time for full process: " << process_time << "milliseconds" << std::endl;
	std::cout << "Time to run: " << run_time << "milliseconds" << std::endl;	
	return 0;

}
