#include "ParseData.h"
#include <regex>
#define TEST_PARSE_MODE

void Parse_MNIST_Training_Data(char* filename,int** pp_training_data,int* p_training_classifications,int num_training_sets,int num_pixels_per_ts)
{
	FILE* fp_training_data = fopen(filename,"rt");
	char* temp_buffer = (char*) malloc(8192*sizeof(char));
	fgets(temp_buffer,8192,fp_training_data);

	for(int i_pattern=0;i_pattern<num_training_sets;i_pattern++)
	{
		fscanf(fp_training_data,"%d,",&p_training_classifications[i_pattern]);
		for(int i_pixel=0;i_pixel<num_pixels_per_ts;i_pixel++)
		{
			fscanf(fp_training_data,"%d,",&pp_training_data[i_pattern][i_pixel]);
		}
#ifdef TEST_PARSE_MODE
		if(i_pattern%1000==0)
			cout << i_pattern << endl;
#endif
	}
	free(temp_buffer);
	fclose(fp_training_data);
}

void Parse_MNIST_Test_Data(char* filename,int** pp_test_data,int num_test_sets,int num_pixels_per_ts)
{
	FILE* fp_test_data = fopen(filename,"rt");
	char* temp_buffer = (char*) malloc(8192*sizeof(char));
	fgets(temp_buffer,8192,fp_test_data);

	for(int i_pattern=0;i_pattern<num_test_sets;i_pattern++)
	{
		for(int i_pixel=0;i_pixel<num_pixels_per_ts;i_pixel++)
		{
			fscanf(fp_test_data,"%d,",&pp_test_data[i_pattern][i_pixel]);
		}
#ifdef TEST_PARSE_MODE
		if(i_pattern%1000==0)
			cout << i_pattern << endl;
#endif
	}
	free(temp_buffer);
	fclose(fp_test_data);
}
