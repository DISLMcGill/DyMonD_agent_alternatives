Training:

Run C_API/training_pb.py to obtain 5 .pb extention models. This script will also test each model, and generate performance metrics on the test data in TestNoSLCor.csv. The data the model is trained on is in TrainNoSLCor.csv: each line is a series of test data, and the corresponding correct prediction. Select the model with the f1 score closest to the average of the 5 models.

Note: this process takes multiple hours, and is faster if run on google colab with the GPU setting.

Building Tensorflow C++ API:

From: https://github.com/FloopCZ/tensorflow_cc
1) Install requirements as necessary from the link
2) git clone https://github.com/FloopCZ/tensorflow_cc.git && cd tensorflow_cc
3) cd tensorflow_cc
4) mkdir build && cd build
5) cmake ..  (this will take a long time)
6) make      (this will take a long time)
7) sudo make install
8) if windows: sudo ldconfig, else, don't do this

Free disk space (optional)
1) rm -rf ~/.cache
2) cd .. && rm -rf build

Running the program:
1) copy example.cpp into tensorflow_cc/tensorflow_cc
2) copy CMakeLists.txt into tensorflow_cc/tensorflow_cc
   - change the location of protobuf in CMakeLists.txt to match where it is installed
3) Also copy in the directory with the saved model into tensorflow_cc/tensorflow_cc and tensorflow_cc/tensorflow_cc/build, and the file test_no_labels.csv in data.zip
4) in the tensorflow_cc/tensorflow_cc/build directory, run
        i) cmake ..
        ii) make
        iii) ./example
This will generate a file called output_predictions.txt.

5) run python3 process_other_results.py to generate the performance metrics of the predictions generated

Notes:
- This runs with tensorflow2.9.0
- Need protobuf version 3.9.1
- May need to run the following commands first to avoid errors in Tensorflow C++ API build
        pip install -U pip numpy wheel packaging requests opt_einsum
        pip install -U keras_preprocessing --no-deps
- To build tensorflow from source on your computer, follow the steps in the link provided above
