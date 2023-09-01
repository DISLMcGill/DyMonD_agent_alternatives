C API:
----------------------------------------------------------------------------------
Training:

Run Data/training_in_python.py to obtain 5 .pb extention models. Select the model with the f1 score closest to the average of the 5 models.

Note: this process takes multiple hours, and is faster if run on google colab with the GPU setting.  

Note on Model format:

Savemodel format will create a directory with 3 subdirectories, including a folder with the .pb model. Only the path to the parent directory is needed in c_api.c

Testing:

1) Download tensorflow ( v2.13 used ), and make sure you have the compatible python version (on the tensorflow website).
Most versions of tensorflow 2.xx can be used if 2.13 not preferred.

2) In c_api.c, replace "Models/model_0" with the path to the folder containing whichever model you wish to use.

3) Run : gcc -I<path to libtensorflow/include> -L<path to libtensorflow/lib> c_api.c -ltensorflow -o main.out
  
  ex: gcc -I../libtensorflow/include/ -L../libtensorflow/lib c_api.c -ltensorflow -o <choose name of executable>
  -> c_api.c contains the code to process the model, generate and output predictions

4) Run ./<name of executable>
This should result in a file called ten_classes_0_results.csv (or whatever you change it to -> change this in c_api.c).

5) Run python3 process_results.py for performance metrics (will need to alter name of ten_classes_0_results.csv in this file if a different name was used.


