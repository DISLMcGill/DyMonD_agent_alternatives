Keras2C:
-------------------------

Training:
Run Mod10Labels-train.py to obtain 5 .h5 models. This script will also test each model, and generate performance metrics on the test data in TestNoSLCor.csv. The data the model is trained on is in TrainNoSLCor.csv: each line is a series of test data, and the corresponding correct prediction. Select the model with the f1 score closest to the average of the 5 models.

Notes:
1)TrainNoSLCor.csv, TestNoSLCor.csv not currently uploaded b/c too big
2) CNN_copy_test_suite.c (the test suite created by step 3 of Testing) is also too large to upload here 

Testing:
1) git clone https://github.com/f0uriest/keras2c
2) Using the automatic test suite:
Run the following command in the parent directory of keras2c to generate files containing layer weights and a test suite:
  python3 -m keras2c -m -t <number of random tests you want> <path to trained model> <name of model>
  
  ex: python3 -m keras2c -m -t 5643 ../ModelAndTestData-NonSecuredTraces/NoSL10L_CNNmodel1.h5 CNNmodelh1
  -> NoSL10L_CNNmodel1.h5 is the name of the model trained in the Training step

To then run the automatic test suite, and obtain the maximum absolute error between the original python model's results and the results from the converted C model on the random data, run:
  i) gcc -std=c99 -I. /include/ -o executable_name model_name.c model_name_test_suite.c -L./include/ /path/to/keras2c/include/libkeras2c.a -lm
  ii) ./executable_name

3) To create and run a test suite to generate predictions from tests in TestNoSLCor.csv, and compare to real values:
  i) python3 create_testsuite.py
  ii) gcc -std=c99 -I. /include/ -o executable_name model_name.c CNN_copy_test_suite.c -L./include/ /path/to/keras2c/include/libkeras2c.a -lm
  iii) ./executable_name
  
  (now, the predictions are in august.txt)
  iv) python3 process_predictions.py
	(enter results file: august.txt)

