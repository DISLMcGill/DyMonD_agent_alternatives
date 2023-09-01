import csv

tests = open("CNN_copy_test_suite.c", "w")
tests.write('#include <stdio.h>')
tests.write("\n")
tests.write(' #include <math.h>')
tests.write("\n")
tests.write(' #include <time.h>')
tests.write("\n")
tests.write(' #include "./include/k2c_include.h" ')
tests.write("\n")
tests.write(' #include "CNNmodelh1.h"')
tests.write("\n")
tests.write("float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2);" )
tests.write("\n")

tests.write("struct timeval GetTimeStamp();")
tests.write("\n")
# open files

data = open('../Data/TestNoSLCor.csv', 'r')
predictions = open('python_predictions1.csv','r')

# initiate readers/writer

data_reader = csv.reader(data)
p_reader = csv.reader(predictions)
writer = csv.writer(tests)

num = 1

# iterate through rows, write tests and predictions

for row1, row2 in zip(data_reader, p_reader):
    tests.write("float test" + str(num) + "_conv2d_3_input_input_array[3600] = {" )
    
    new_row = []
    old_row = row1[:-1]
    for x in range(len(old_row)):
        new_row.append(float(old_row[x])/255.0)
    writer.writerow(new_row)

    tests.write("};")

    tests.write("k2c_tensor test"+ str(num) +"_conv2d_3_input_input = {&test"+ str(num) +"_conv2d_3_input_input_array[0],3,3600,{100,  6,  6,  1,  1}};")

    tests.write("float keras_dense_4_test" + str(num) + "_array[9] = {")
    writer.writerow(row2)

    tests.write("};")
	
    tests.write("k2c_tensor keras_dense_4_test"+ str(num) +" = {&keras_dense_4_test"+ str(num) +"_array[0],1,9,{9,1,1,1,1}}; float c_dense_4_test" + str(num) + "_array[9] = {0}; k2c_tensor c_dense_4_test"+ str(num) +" = {&c_dense_4_test"+ str(num) +"_array[0],1,9,{9,1,1,1,1}};")

    num = num + 1


tests.write("int main(){ \n"
	"float errors[5643]; \n"
	"size_t num_tests = 5643; \n"
	"size_t num_outputs = 1; \n"
	"float* conv2d_3_output_array; \n"
	"float* conv2d_3_padded_input_array;\n"
	"float* conv2d_3_kernel_array;\n"
	"float* conv2d_3_bias_array;\n"
	"float* batch_normalization_3_output_array;\n"
	"float* batch_normalization_3_mean_array;\n"
	"float* batch_normalization_3_stdev_array;\n"
	"float* batch_normalization_3_gamma_array;\n"
	"float* batch_normalization_3_beta_array;\n"
	"float* conv2d_4_output_array;\n"
	"float* conv2d_4_padded_input_array;\n"
	"float* conv2d_4_kernel_array;\n"
	"float* conv2d_4_bias_array;\n"
	"float* batch_normalization_4_output_array;\n"
	"float* batch_normalization_4_mean_array;\n"
	"float* batch_normalization_4_stdev_array;\n"
	"float* batch_normalization_4_gamma_array;\n"
	"float* batch_normalization_4_beta_array;\n"
	"float* time_distributed_3_output_array;\n"
	"float* max_pooling1d_2_output_array;\n"
	"float* max_pooling1d_2_timeslice_input_array;\n"
	"float* max_pooling1d_2_timeslice_output_array;\n"
	"float* time_distributed_4_output_array;\n"
	"float* flatten_2_output_array;\n"
	"float* flatten_2_timeslice_input_array;\n"
	"float* flatten_2_timeslice_output_array;\n"
	"float* backward_lstm_2_output_array;\n"
	"float* backward_lstm_2_kernel_array;\n"
	"float* backward_lstm_2_recurrent_kernel_array;\n"
	"float* backward_lstm_2_bias_array;\n"
	"float* forward_lstm_2_output_array;\n"
	"float* forward_lstm_2_kernel_array;\n"
	"float* forward_lstm_2_recurrent_kernel_array;\n"
	"float* forward_lstm_2_bias_array;\n"
	"float* bidirectional_2_output_array;\n"
	"float* dense_3_output_array;\n"
	"float* dense_3_kernel_array;\n"
	"float* dense_3_bias_array;\n"
	"float* dense_4_kernel_array;\n"
	"float* dense_4_bias_array;\n")

tests.write("CNNmodelh1_initialize(&conv2d_3_output_array,&conv2d_3_padded_input_array,&conv2d_3_kernel_array,&conv2d_3_bias_array,&batch_normalization_3_output_array,&batch_normalization_3_mean_array,&batch_normalization_3_stdev_array,&batch_normalization_3_gamma_array,&batch_normalization_3_beta_array,&conv2d_4_output_array,&conv2d_4_padded_input_array,&conv2d_4_kernel_array,&conv2d_4_bias_array,&batch_normalization_4_output_array,&batch_normalization_4_mean_array,&batch_normalization_4_stdev_array,&batch_normalization_4_gamma_array,&batch_normalization_4_beta_array,&time_distributed_3_output_array,&max_pooling1d_2_output_array,&max_pooling1d_2_timeslice_input_array,&max_pooling1d_2_timeslice_output_array,&time_distributed_4_output_array,&flatten_2_output_array,&flatten_2_timeslice_input_array,&flatten_2_timeslice_output_array,&backward_lstm_2_output_array,&backward_lstm_2_kernel_array,&backward_lstm_2_recurrent_kernel_array,&backward_lstm_2_bias_array,&forward_lstm_2_output_array,&forward_lstm_2_kernel_array,&forward_lstm_2_recurrent_kernel_array,&forward_lstm_2_bias_array,&bidirectional_2_output_array,&dense_3_output_array,&dense_3_kernel_array,&dense_3_bias_array,&dense_4_kernel_array,&dense_4_bias_array); clock_t t0 = clock();")


for i in range (1,5643):
	tests.write("CNNmodelh1(&test"+ str(i) + "_conv2d_3_input_input,&c_dense_4_test" + str(i) + ",conv2d_3_output_array,conv2d_3_padded_input_array,conv2d_3_kernel_array,conv2d_3_bias_array,batch_normalization_3_output_array,batch_normalization_3_mean_array,batch_normalization_3_stdev_array,batch_normalization_3_gamma_array,batch_normalization_3_beta_array,conv2d_4_output_array,conv2d_4_padded_input_array,conv2d_4_kernel_array,conv2d_4_bias_array,batch_normalization_4_output_array,batch_normalization_4_mean_array,batch_normalization_4_stdev_array,batch_normalization_4_gamma_array,batch_normalization_4_beta_array,time_distributed_3_output_array,max_pooling1d_2_output_array,max_pooling1d_2_timeslice_input_array,max_pooling1d_2_timeslice_output_array,time_distributed_4_output_array,flatten_2_output_array,flatten_2_timeslice_input_array,flatten_2_timeslice_output_array,backward_lstm_2_output_array,backward_lstm_2_kernel_array,backward_lstm_2_recurrent_kernel_array,backward_lstm_2_bias_array,forward_lstm_2_output_array,forward_lstm_2_kernel_array,forward_lstm_2_recurrent_kernel_array,forward_lstm_2_bias_array,bidirectional_2_output_array,dense_3_output_array,dense_3_kernel_array,dense_3_bias_array,dense_4_kernel_array,dense_4_bias_array); ")

tests.write("clock_t t1 = clock();")
#tests.write('printf("Average time over 5643 tests: %e s", ((double)t1-t0)/(double)CLOCKS_PER_SEC/(double)5643);')
tests.write('FILE *file = fopen("c_model_predictions.txt", "w");')
for i in range(5643):
	tests.write("errors[" + str(i) + "] = maxabs(&keras_dense_4_test" + str(i+1) + ",&c_dense_4_test" + str(i+1) + ");" + "\n")
	tests.write('fprintf(file, "%f %f %f %f %f %f %f %f %f ", c_dense_4_test' + str(i+1) + '_array[0],c_dense_4_test' + str(i+1) + '_array[1],c_dense_4_test' + str(i+1) + '_array[2],c_dense_4_test' + str(i+1) + '_array[3], c_dense_4_test' + str(i+1) + '_array[4], c_dense_4_test' + str(i+1) + '_array[5], c_dense_4_test' + str(i+1) + '_array[6], c_dense_4_test' + str(i+1) + '_array[7], c_dense_4_test' + str(i+1) + '_array[8]);')
	

tests.write('fclose(file);')
tests.write("float maxerror = errors[0];" + "\n" + "for(size_t i=1; i< num_tests*num_outputs;i++){" + "\n" + "if (errors[i] > maxerror) {" + "\n" + "maxerror = errors[i];}}" + "\n")

tests.write('printf("Max error : %f",maxerror);') 

tests.write("CNNmodelh1_terminate(conv2d_3_output_array,conv2d_3_padded_input_array,conv2d_3_kernel_array,conv2d_3_bias_array,batch_normalization_3_output_array,batch_normalization_3_mean_array,batch_normalization_3_stdev_array,batch_normalization_3_gamma_array,batch_normalization_3_beta_array,conv2d_4_output_array,conv2d_4_padded_input_array,conv2d_4_kernel_array,conv2d_4_bias_array,batch_normalization_4_output_array,batch_normalization_4_mean_array,batch_normalization_4_stdev_array,batch_normalization_4_gamma_array,batch_normalization_4_beta_array,time_distributed_3_output_array,max_pooling1d_2_output_array,max_pooling1d_2_timeslice_input_array,max_pooling1d_2_timeslice_output_array,time_distributed_4_output_array,flatten_2_output_array,flatten_2_timeslice_input_array,flatten_2_timeslice_output_array,backward_lstm_2_output_array,backward_lstm_2_kernel_array,backward_lstm_2_recurrent_kernel_array,backward_lstm_2_bias_array,forward_lstm_2_output_array,forward_lstm_2_kernel_array,forward_lstm_2_recurrent_kernel_array,forward_lstm_2_bias_array,bidirectional_2_output_array,dense_3_output_array,dense_3_kernel_array,dense_3_bias_array,dense_4_kernel_array,dense_4_bias_array); if (maxerror > 1e-05) {return 1;}return 0;}" )

tests.write("float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2){" + "\n" + "float x = 0;" + "\n" + "float y = 0;" + "\n" + "for(size_t i=0; i<tensor1->numel; i++){" + "\n" + "y = fabsf(tensor1->array[i]-tensor2->array[i]);" + "\n" + "if (y>x) {x=y;}}" + "\n" + "return x;}" + "\n" )


tests.close()
data.close()
predictions.close()

