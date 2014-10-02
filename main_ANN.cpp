#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <set>

#include "ParseData.h"

#include "ANN.h"

using namespace std;
const int NUM_PATTERNS=42000;
const int NUM_PIXELS_PER_PATTERN=784;
const int NUM_CLASSIFICATIONS = 10;  // how many classes are there ?

CNeuralNetwork<double> neural_net;

// convert the (initially integer-valued features) into doubles and scale them....

void FeatureScaling(int** initial_values,double** scaled_values,int* initial_classifications,double** conv_classifications,int num_datasets,int num_features);

void FeatureScaling(int** initial_values,double** scaled_values,int* initial_classifications,double** conv_classifications,int num_datasets,int num_features)
{
	double mean;
	for(int index_dimension=0;index_dimension<num_features;index_dimension++)
	{
		// get the statistics for the current
		// dimensions
		int min =  initial_values[0][index_dimension];
		int max =  initial_values[0][index_dimension];
		double mean=0.0f;

		//curr_set.clear();
		
		for(int index_pattern=0;index_pattern<num_datasets;index_pattern++)
		{
			mean+=initial_values[index_pattern][index_dimension];
			
			if(initial_values[index_pattern][index_dimension]<min)
				min = initial_values[index_pattern][index_dimension];

			if(initial_values[index_pattern][index_dimension]>max)
				max = initial_values[index_pattern][index_dimension];
		}
		mean/=num_datasets;
		double range = (double)(max-min);

		// now scale the values....
		for(int index_pattern=0;index_pattern<num_datasets;index_pattern++)
		{
			scaled_values[index_pattern][index_dimension] = (double)initial_values[index_pattern][index_dimension]-mean;
			if(range!=0.0f)
			{
				scaled_values[index_pattern][index_dimension]/=range;
			}
		}
		
		


		// get the statistics
/*		cout << "current dimension: " << index_dimension << endl;
		cout << "mean value: " << mean << endl;
		cout << "Minimum: " << min << endl;
		cout << "Maximum: " << max << endl;
		cout << "Range: " << range << endl;*/

	}
	// now update the classifications...
	for(int index_pattern=0;index_pattern<num_datasets;index_pattern++)
	{
		conv_classifications[index_pattern][initial_classifications[index_pattern]]=1.0f;
		/*if(index_pattern%200==0)
		{
			// give out some examples:
			cout << "classification for pattern: " << index_pattern << " : " << initial_classifications[index_pattern] << endl;
			cout << "corresponding classification vector: " << endl;
			for(int i=0;i<NUM_CLASSIFICATIONS;++i)
				cout << conv_classifications[index_pattern][i] << "\t";
			cout << endl;

			// now print out the corresponding pattern:
			for(int index_pixel=0;index_pixel<num_features;index_pixel++)
			{
				cout << scaled_values[index_pattern][index_pixel] << endl;
			}
		}*/
	}

}


void main()
{
	char* pathname_train = new char[256];
	strcpy(pathname_train,"C:\\Studium\\DataMining\\MNIST\\train.csv");

	// reserve space..
	int** g_training_data;
	int* g_training_classifications;
	
	g_training_data = new int*[NUM_PATTERNS];
	g_training_classifications  = new int[NUM_PATTERNS];
	
	double** d_training_data = new double*[NUM_PATTERNS];
	double** d_training_classifications = new double*[NUM_PATTERNS];


	for(int i=0;i<NUM_PATTERNS;++i)
	{
		g_training_data[i] = new int[NUM_PIXELS_PER_PATTERN];
		d_training_data[i] = new double[NUM_PIXELS_PER_PATTERN];
		d_training_classifications[i] = new double[NUM_CLASSIFICATIONS]();
	}
	

	cout << "Parsing Data" << endl;
	Parse_MNIST_Training_Data(pathname_train,g_training_data,g_training_classifications,10000,NUM_PIXELS_PER_PATTERN);

	cout << "Performing Feature Scaling" << endl;
	FeatureScaling(g_training_data,d_training_data,g_training_classifications,d_training_classifications,10000,NUM_PIXELS_PER_PATTERN);
	cout << "Building ANN: " << endl;
	srand(time(NULL));
	int num_layers = 3; // 2 = input and output layer, 3 = input, 1 hidden and output layer...
	int* nn_per_layer = new int[3];
	
	nn_per_layer[0]=784;
	nn_per_layer[1]=784;
	nn_per_layer[2]=10;
	cout << "Building Neural Network: " << endl;
	neural_net.BuildFCNeuralNet(num_layers,nn_per_layer);
	//TrainNeuralNet(T** input_patterns,int num_input_patterns,int dimension_pattern,double** classifications)
	neural_net.SetBatchSize(100);
	neural_net.SetNumClassifications(10);

	neural_net.SetLearningRate(3e-2f);
	
	for(int i=0;i<7;++i)
		neural_net.TrainNeuralNet(d_training_data,500,NUM_PIXELS_PER_PATTERN,d_training_classifications);
	
	char filename_output[256];
	strcpy(filename_output,"neural_net.txt");
	neural_net.WriteToFile(filename_output);
	/*
	char filename_input[256];
	strcpy(filename_input,"neural_net.txt");

	neural_net.ReadFromFile(filename_input);

	for(int i=0;i<10;++i)
	{
		cout << "pattern: " << i << endl;
		cout << "desired classification: " << g_training_classifications[i] << endl;
		neural_net.Classify(NUM_PIXELS_PER_PATTERN,d_training_data[i]);
	}*/
	//neural_net.PrintCompleteNeuralNetwork();
	/*double* input = new double[3];
	input[0] = 1.0f;
	input[1] = 2.0f;*/

	//input[2] = 2.0f;
	//cout << "calculating output for one example: " << endl;
	//vector<double> ausgabe=neural_net.ForwardPass(2,d_training_data[0]);
	//cout << "NN output: " << endl;
	//for(int i=0;i<10;++i)
	//	cout << ausgabe[i] << endl;
	//cout << "After 1 Forward Pass: " << endl;
	//neural_net.PrintCompleteNeuralNetwork();
	cout << "Eingabe zum Beenden: " << endl;
	int final_char;
	cin >> final_char;
	delete[] pathname_train;
	for(int i=0;i<NUM_PATTERNS;++i)
	{
		delete[] g_training_data[i];
		delete[] d_training_data[i];
		delete[] d_training_classifications[i];
	}
	delete[] d_training_data;
	delete[] d_training_classifications;

	delete[] g_training_classifications;
	delete[] g_training_data;

}