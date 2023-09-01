C API:
----------------------------------------------------------------------------------
Training:

Run training_in_python.py to obtain 5 .pb extention models. This script will also test each model, and generate performance metrics on the test data in TestNoSLCor.csv. The data the model is trained on is in TrainNoSLCor.csv: each line is a series of test data, and the corresponding correct prediction. Select the model with the f1 score closest to the average of the 5 models.

Note: this process takes multiple hours, and is faster if run on google colab with the GPU setting.  

Note on Model format:

Savemodel format will create a directory with 3 subdirectories, including a folder with the .pb model. Only the path to the parent directory is needed in c_api.c

Testing:
** what you will need to download:
- tensorflow ( v2.13 used ), and make sure you have the compatible python version (on the tensorflow website)
- numpy 
1) In c_api.c, replace "Models/model_0" with whichever model you wish to use

2) Run : gcc -I<path to libtensorflow/include> -L<path to libtensorflow/lib> c_api.c -ltensorflow -o main.out
  ex: gcc -I../libtensorflow/include/ -L../libtensorflow/lib file6.c -ltensorflow -o <choose name of executable>

3) Run ./<name of executable>
This should result in a file called ten_classes_0_results.csv (or whatever you change it to)

4) Run python3 process_results.py for performance metrics


