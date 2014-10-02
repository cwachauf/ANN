#include <iostream>
#include <stdio.h>
using namespace std;

// Parse MNIST Training Data in
// .csv - format
// Parameters are: 
// char* filename --> self-explanatory (filename, including full path to training-data file)
// pp_training_data --> 2-dimensional integer-array that will be filled with the training-data set
// p_training_classifications --> integer-array that will be filled with the classifications (0 to 9)
// arrays need to be allocated !!!
void Parse_MNIST_Training_Data(char* filename,int** pp_training_data,int* p_training_classifications,int num_training_sets,int num_pixels_per_ts);
void Parse_MNIST_Test_Data(char* filename,int** pp_test_data,int num_test_sets,int num_pixels_per_ts);
