#include <iostream>
#include <vector>
#include <map>
#include <stdlib.h>
#include <math.h>

using namespace std;

template <typename T>
T logistic(T x)
{
	return (1.0f/(1.0f+exp(-x)));
}

template <typename T>
T neuron_function(T x)
{
	return logistic(x);
}

template<typename T>
class CNeuron
{
private:
	int m_layer; // layer identifier
	int m_neuron_ID; // identifier for this neuron...
	int m_num_connections;
	T m_curr_value; // current value --> for layer==1 neurons this is the input-value...
	double m_bias;			// current value of the bias-weight...
	vector<int> m_indices_parent_nodes; // indices to parent nodes....
	map<int,double> m_weights; // mapping indices of neurons to weights (which weight has the connection from this neuron to one of his children ??)
public:
	void SetNumConnections(int num_connections);
	void SetNeuronID(int neuron_ID);
	void SetWeights(map<int,double> weights);
	void SetLayer(int layer);
	void SetCurrentValue(T curr_value);
	void SetBias(double bias);
	void SetIndicesParentNodes(vector<int> m_indices_parent_nodes);

	
	int GetLayer(){return m_layer;};
	int GetNeuronID(){return m_neuron_ID;};
	int GetNumConnections(){return m_num_connections;};
	double GetBias(){return m_bias;};
	T GetCurrValue(){return m_curr_value;};
	map<int,double>* GetWeights(){return &m_weights;};
	vector<int>* GetIndicesParentNodes(){ return &m_indices_parent_nodes;};

	void Print();
};

template <typename T>
void CNeuron<T>::SetIndicesParentNodes(vector<int> indices_parent_nodes)
{
	m_indices_parent_nodes = indices_parent_nodes;
}

template <typename T>
void CNeuron<T>::Print()
{
	cout << "Neuron ID: " << m_neuron_ID << endl;
	cout << "Layer: " << m_layer << endl;
	cout << "Number Connections: " << m_num_connections << endl;
	cout << "Current Value: " << m_curr_value << endl;

	cout << "Value of Bias-weight: " << m_bias << endl;

	// print out the current weights:
	map<int,double>::iterator it_weights;
	for(it_weights=m_weights.begin();it_weights!=m_weights.end();it_weights++)
	{
		cout << "weight to node: " << it_weights->first << " : " << it_weights->second << endl;
	}
	
	// print out the parent neurons:
	for(int i=0;i<m_indices_parent_nodes.size();i++)
		cout << "index: " << m_indices_parent_nodes[i] << endl;
	cout << endl;
}

template <typename T>
void CNeuron<T>::SetCurrentValue(T curr_value)
{
	m_curr_value = curr_value;
}

template <typename T>
void CNeuron<T>::SetBias(double bias)
{
	m_bias = bias;
}

template <typename T>
void CNeuron<T>::SetWeights(map<int,double> weights)
{
	m_weights=weights;
}

template <typename T>
void CNeuron<T>::SetLayer(int layer)
{
	m_layer = layer;
}

template <typename T>
void CNeuron<T>::SetNumConnections(int num_connections)
{
	m_num_connections=num_connections;
}

template <typename T>
void CNeuron<T>::SetNeuronID(int neuron_ID)
{
	m_neuron_ID = neuron_ID;
}



template<typename T>
class CNeuralNetwork
{
private:
	vector<CNeuron<T> > m_neurons;
	int m_num_layers;
	int* m_num_neurons_per_layer;
	int m_num_total_neurons;
	int m_batch_size;
	int m_num_classifications;
	double m_global_learning_rate; //
	void CalculateDeltas(double* desired_classification,double* real_classification,double* deltas);
	void UpdateWeights(double* deltas);
	vector<double> ForwardPass(int dimension_pattern,T* input_pattern);
public:
	void SetBatchSize(int batch_size){m_batch_size=batch_size;};
	void SetLearningRate(double learning_rate){m_global_learning_rate=learning_rate;};
	void SetNumClassifications(int num_classifications){m_num_classifications=num_classifications;};
	void BuildFCNeuralNet(int num_layers,int* num_neurons_per_layer);
	void TrainNeuralNet(T** input_patterns,int num_input_patterns,int dimension_pattern,double** classifications);
	

	int GetBatchSize(){return m_batch_size;};
	int GetNumClassifications(){return m_num_classifications;};
	void PrintCompleteNeuralNetwork(); // Prints information about the entire neural network...
	int Classify(int dimension_pattern,T* input_pattern);

	double GetLearningRate();
	double CalculateErrorFunction1(double* desired_classification,double* classification,double lambda);
	double CalculateErrorFunction2(double* desired_classification,double* classification);	// simple quadratic error function, without regularization....
	void WriteToFile(char* filename);
	void ReadFromFile(char* filename);
};

template <typename T>
int CNeuralNetwork<T>::Classify(int dimension_pattern,T* input_pattern)
{

	vector<double> result_vector;
	result_vector=ForwardPass(dimension_pattern,input_pattern);
	
	int index_max=0;
	double max = result_vector[0];
	for(int i=0;i<result_vector.size();i++)
	{
		//cout << result_vector[i] << "\t";
		//cout << endl;
		if(result_vector[i]>max)
		{
			index_max = i;
			max = result_vector[i];
		}
	}

	cout << "classification: " << index_max << endl;
	return index_max;
}


template <typename T>
void CNeuralNetwork<T>::WriteToFile(char* filename)
{
	FILE* fp_output = fopen(filename,"wt");

	// write number of layers, neurons per layer and then for
	fprintf(fp_output,"%d\n",m_num_layers);
	for(int i=0;i<m_num_layers;++i)
		fprintf(fp_output,"%d\n",m_num_neurons_per_layer[i]);

	for(int index_neuron=0;index_neuron<m_num_total_neurons;++index_neuron)
	{
		int neuron_ID = m_neurons[index_neuron].GetNeuronID();
		fprintf(fp_output,"%d\n",neuron_ID);
		int neuron_Layer = m_neurons[index_neuron].GetLayer();
		fprintf(fp_output,"%d\n",neuron_Layer);
		double bias_value = m_neurons[index_neuron].GetBias();
		fprintf(fp_output,"%lf\n",bias_value);
		map<int,double>* p_weights = m_neurons[index_neuron].GetWeights();
		int num_weights = p_weights->size();
		fprintf(fp_output,"%d\n",num_weights);
		map<int,double>::iterator it_map;
		for(it_map=p_weights->begin();it_map!=p_weights->end();it_map++)
			fprintf(fp_output,"%d\t%f\n",it_map->first,it_map->second);
		
		vector<int>* indices_incident_neurons = m_neurons[index_neuron].GetIndicesParentNodes();
		int num_indices_parent_nodes = indices_incident_neurons->size();

		fprintf(fp_output,"%d\n",num_indices_parent_nodes);
		for(int index_parent_node=0;index_parent_node<num_indices_parent_nodes;index_parent_node++)
			fprintf(fp_output,"%d\n",(*indices_incident_neurons)[index_parent_node]);
	}
	fclose(fp_output);
}

template <typename T>
void CNeuralNetwork<T>::ReadFromFile(char* filename)
{
	FILE* fp_input = fopen(filename,"rt");

	int num_layers;
	fscanf(fp_input,"%d\n",&num_layers);
	this->m_num_layers=num_layers;
	cout << "network has: " << num_layers << " layers" << endl;
	
	this->m_num_neurons_per_layer = new int[m_num_layers];
	int total_num_neurons = 0;
	for(int index_layer=0;index_layer<num_layers;index_layer++)
	{

		int curr_num_layers;
		fscanf(fp_input,"%d\n",&curr_num_layers);
		total_num_neurons+=curr_num_layers;
		m_num_neurons_per_layer[index_layer] = curr_num_layers;
	}
	m_num_total_neurons = total_num_neurons;
	// read in the information for each individual neuron...
	m_neurons.reserve(m_num_total_neurons);
	for(int index_neuron=0;index_neuron<total_num_neurons;index_neuron++)
	{
		CNeuron<T> temp_neuron;
	
		if(index_neuron%100==0)
			cout << "reading neuron: " << index_neuron << endl;
		int neuron_id;
		int neuron_layer;
		fscanf(fp_input,"%d\n",&neuron_id);
		fscanf(fp_input,"%d\n",&neuron_layer);
		double bias_value;
		fscanf(fp_input,"%lf\n",&bias_value);
		temp_neuron.SetBias(bias_value);
		int num_weights;
		fscanf(fp_input,"%d\n",&num_weights);

		temp_neuron.SetNeuronID(neuron_id);
		temp_neuron.SetLayer(neuron_layer);
		temp_neuron.SetNumConnections(num_weights);
		
		map<int,double> temp_weights;
		for(int index_weight=0;index_weight<num_weights;index_weight++)
		{
			int index_next_node;
			double value_of_weight;
			fscanf(fp_input,"%d\t%lf",&index_next_node,&value_of_weight);
			temp_weights[index_next_node] = value_of_weight;
		}
		int num_incident_weights;
		fscanf(fp_input,"%d\n",&num_incident_weights);
		vector<int> indices_incident_neurons;
		indices_incident_neurons.reserve(num_incident_weights);
		for(int index_incident_neurons=0;index_incident_neurons<num_incident_weights;index_incident_neurons++)
		{
			int index_incident_neuron;
			fscanf(fp_input,"%d\n",&index_incident_neuron);
			indices_incident_neurons.push_back(index_incident_neuron);
		}
		temp_neuron.SetWeights(temp_weights);
		temp_neuron.SetIndicesParentNodes(indices_incident_neurons);
		this->m_neurons.push_back(temp_neuron);

	}




	fclose(fp_input);
}

template <typename T>
void CNeuralNetwork<T>::UpdateWeights(double* deltas)
{
	// go layer by layer, starting with the output layer,
	// finishing with (inclusive) the first inner layer..

	int curr_offset=m_num_total_neurons;
	for(int curr_layer=m_num_layers-1;curr_layer>0;curr_layer--)
	{
		curr_offset-=m_num_neurons_per_layer[curr_layer];
		// inner loop over all neurons of this layer..
		for(int i=0;i<m_num_neurons_per_layer[curr_layer];i++)
		{
			int curr_index_neuron = curr_offset+i;

			// update the bias units first:
			double delta_bias_weight = 0.0f;
			delta_bias_weight = -m_global_learning_rate*(deltas[curr_index_neuron]);

			double curr_bias_weight = m_neurons[curr_index_neuron].GetBias();
			curr_bias_weight+=delta_bias_weight;
			m_neurons[curr_index_neuron].SetBias(curr_bias_weight);


			// get the incident weights...
			vector<int>* p_incident_neurons = m_neurons[curr_index_neuron].GetIndicesParentNodes();
			for(int k=0;k<p_incident_neurons->size();++k)
			{
				map<int,double>* p_weights = m_neurons[(*p_incident_neurons)[k]].GetWeights();
				map<int,double>::const_iterator it_map;
				it_map = p_weights->find(curr_index_neuron);
				if(it_map==p_weights->end())
					cout << "SHIT, ERROR; WEIGHT NOT FOUND!" << endl;
				else
				{
					double curr_weight = it_map->second;
					double x = m_neurons[(*p_incident_neurons)[k]].GetCurrValue();
					double delta_weight = -m_global_learning_rate*(deltas[curr_index_neuron])*x;
					curr_weight+=delta_weight;
					(*p_weights)[curr_index_neuron]=curr_weight;
				}
			}

//			map<int,double>* GetWeights(){return &m_weights;};
	//vector<int>* GetIndicesParentNodes(){ return &m_indices_parent_nodes;};
			// now update the rest of the neuron-weights...
		}
	}

}

template <typename T>
double CNeuralNetwork<T>::GetLearningRate()
{
	return m_global_learning_rate;
}

template <typename T>
double CNeuralNetwork<T>::CalculateErrorFunction2(double* desired_classification,double* classification)
{
	double total_error=0.0f;
	for(int i=0;i<m_num_classifications;++i)
		total_error+=(desired_classification[i]-classification[i])*(desired_classification[i]-classification[i]);
	return 0.5f*total_error;
}

template <typename T>
double CNeuralNetwork<T>::CalculateErrorFunction1(double* desired_classification,double* classification,double lambda)
{
	double total_error=0.0f;
	for(int k=0;k<m_num_classifications;++k)
		total_error-=(desired_classification[k]*log(classification[k]))+(1.0f-desired_classification[k])*(log(1.0f-classification[k]));
	
	// add the regularization term...
	int curr_index=0;
	for(int index_layer=0;index_layer<m_num_layers-1;index_layer++)
	{
		for(int index_neuron=0;index_neuron<m_num_neurons_per_layer[index_layer];index_neuron++)
		{
			//map<int,double> m_weights;
			// 
			map<int,double>* p_weights = m_neurons[curr_index].GetWeights();
			
			map<int,double>::iterator it_map;
			for(it_map=p_weights->begin();it_map!=p_weights->end();it_map++)
				total_error+=lambda/2.0f*it_map->second*it_map->second;

			curr_index++;
		}
	}
	return total_error;
}

// see wikipedia:
// en.wikipedia.org/wiki/Backpropagation
template <typename T>
void CNeuralNetwork<T>::CalculateDeltas(double* desired_classification,double* real_classification,double* deltas)
{
	// first calculate the delta-values of the output-neurons:
	
	// index of first output-neuron...
	int curr_index = m_num_total_neurons-m_num_neurons_per_layer[m_num_layers-1];
	double o_j;
	double t_j;
	double phi_j;

	for(int j=0;j<m_num_neurons_per_layer[m_num_layers-1];++j)
	{
		o_j = real_classification[j];
		t_j = desired_classification[j];
		phi_j = o_j; // in this case ?!?!
		deltas[curr_index]=(o_j-t_j)*phi_j*(1.0f-phi_j);	
		curr_index++;
	}

	// now calculate the deltas for the inner neurons....
	//	curr_index = m_num_total_neurons-m_num_neurons_per_layer[m_num_layers-1]-1;

	int curr_offset = m_num_total_neurons-m_num_neurons_per_layer[m_num_layers-1];

	for(int index_layer=m_num_layers-2;index_layer>=0;index_layer-=1)
	{
		curr_offset-=m_num_neurons_per_layer[index_layer];

		for(int i=0;i<m_num_neurons_per_layer[index_layer];++i)
		{
			int j = curr_offset+i;

			phi_j = m_neurons[j].GetCurrValue();
			map<int,double>* p_weights=m_neurons[j].GetWeights();
			double inner_sum=0;
			map<int,double>::iterator iter;
			for(iter=p_weights->begin();iter!=p_weights->end();iter++)
				inner_sum+=deltas[iter->first]*iter->second;
			deltas[j] = inner_sum*phi_j*(1.0f-phi_j);
		}
	}
}

template <typename T> 
void CNeuralNetwork<T>::TrainNeuralNet(T** input_patterns,int num_input_patterns,int dimension_pattern,double** classifications)
{
	// calculate again the total number of neurons...

	int total_num_neurons =0;
	for(int i=0;i<m_num_layers;++i)
		total_num_neurons+=m_num_neurons_per_layer[i];

	double* temp_delta_values = new double[total_num_neurons];

	for(int index_pattern=0;index_pattern<num_input_patterns;index_pattern++)
	{
		cout << "index of pattern: " << index_pattern << endl;
		cout << "output is supposed to be: " << endl;
		for(int index_classification=0;index_classification<m_num_classifications;index_classification++)
			cout << classifications[index_pattern][index_classification] << "\t";
		cout << endl;

		cout << "we obtain: " << endl;
		vector<double> result = ForwardPass(dimension_pattern,input_patterns[index_pattern]);
		for(int index_classification=0;index_classification<m_num_classifications;index_classification++)
		{
			cout << result[index_classification] << "\t";
		}

		// calculate error
		double error = CalculateErrorFunction1(classifications[index_pattern],&result[0],0.0001f);
		cout << "error of this classification: " << error << endl;

		double error2 = CalculateErrorFunction2(classifications[index_pattern],&result[0]);
		cout << "error2 of this classification: " << error2 << endl;
		
		// calculate the delta values...
		CalculateDeltas(classifications[index_pattern],&result[0],temp_delta_values);
		// update the weights...
		UpdateWeights(temp_delta_values);

		cout << endl;
	}
	delete[] temp_delta_values;
}


template<typename T>
void CNeuralNetwork<T>::PrintCompleteNeuralNetwork()
{
	cout << "Number of Layers: " << m_num_layers << endl;
	
	for(int i=0;i<m_num_layers;++i)
		cout << "Number of Nodes in Layer: " << i << " ; " << m_num_neurons_per_layer[i] << endl;

	// print out the neuron information for each neuron...
	int curr_index = 0;
	for(int i=0;i<m_num_layers;++i)
		for(int j=0;j<m_num_neurons_per_layer[i];j++)
		{
			m_neurons[curr_index].Print();
			curr_index++;
		}
}

template<typename T>
vector<double> CNeuralNetwork<T>::ForwardPass(int dimension,T* pattern)
{
	// calculate the values layer for layer and neuron for neuron...

	// input layer:
	// values are simply assigned...
	for(int i=0;i<m_num_neurons_per_layer[0];++i)
	{
		int layer=m_neurons[i].GetLayer();
		if(layer!=0)
		{
			cout << "FEHLER..." << endl;
		}
		else
		{
			m_neurons[i].SetCurrentValue(pattern[i]);
		}
	}
	vector<double> output;
	int curr_index = m_num_neurons_per_layer[0];
	// all the other layers: do the math...
	for(int index_layer=1;index_layer<m_num_layers;index_layer++)
	{
		for(int index_neuron=0;index_neuron<m_num_neurons_per_layer[index_layer];index_neuron++)
		{
			// get the incident neurons of the current neuron
			vector<int>* indices_incident_neurons = m_neurons[curr_index].GetIndicesParentNodes();
			int num_incident_neurons = (*indices_incident_neurons).size();
			double total_value=0.0f;
			for(int index_incident_neuron=0;index_incident_neuron<num_incident_neurons;index_incident_neuron++)
			{
				map<int,double>* map_current_incident_node = m_neurons[(*indices_incident_neurons)[index_incident_neuron]].GetWeights();
				double curr_weight = (*map_current_incident_node)[curr_index];
				double curr_value = m_neurons[(*indices_incident_neurons)[index_incident_neuron]].GetCurrValue();
				total_value+=curr_weight*curr_value;
			}
			// add the bias weight..
			total_value+=m_neurons[curr_index].GetBias();
			// set the new value of the 
			total_value=neuron_function(total_value);
			m_neurons[curr_index].SetCurrentValue(total_value);
			if(index_layer==m_num_layers-1) // output layer reached
			{
				output.push_back(total_value);
			}
			///if(curr_index%10==0)
		//	cout << curr_index << endl;
			curr_index++;
		}
	}
	return output;

	// the output are the "values" of the output layer..
	
}

template<typename T>
void CNeuralNetwork<T>::BuildFCNeuralNet(int num_layers,int* num_neurons_per_layer)
{
	m_neurons.clear();

	m_num_layers=num_layers;
	m_num_neurons_per_layer = new int[m_num_layers];
	int total_num_neurons = 0;
	for(int i=0;i<m_num_layers;++i)
	{
		m_num_neurons_per_layer[i]=num_neurons_per_layer[i];
		total_num_neurons+=num_neurons_per_layer[i];
	}
	m_num_total_neurons=total_num_neurons;
	m_neurons.reserve(total_num_neurons);

	cout << "total number of neurons: " << total_num_neurons << endl;

	CNeuron<T> template_neuron;	
	int ID=0;
	int sum_of_all_previous_layers=0;
	for(int i=0;i<m_num_layers-1;i++)
	{
		template_neuron.SetLayer(i);
		template_neuron.SetNumConnections(m_num_neurons_per_layer[i+1]); // one connection to each neuron of the next layer...
		sum_of_all_previous_layers+=m_num_neurons_per_layer[i];

		for(int j=0;j<m_num_neurons_per_layer[i];j++)
		{
			template_neuron.SetNeuronID(ID);
			
			map<int,double> weight_values;
			for(int k=0;k<m_num_neurons_per_layer[i+1];k++)
			{
				weight_values[sum_of_all_previous_layers+k] = (rand()%1000)*0.001f-0.5f;
			}
			template_neuron.SetWeights(weight_values);
			
			double rand_bias_weight = (rand()%1000)*0.001f-0.5f;
			template_neuron.SetBias(rand_bias_weight);
			template_neuron.SetCurrentValue(T(0));

			m_neurons.push_back(template_neuron);
			if(ID%10==0)
				cout << ID << endl;
			ID+=1;
		}
	}
	// neurons in the last layer
	for(int j=0;j<m_num_neurons_per_layer[m_num_layers-1];j++)
	{
		template_neuron.SetLayer(m_num_layers-1);
		template_neuron.SetNumConnections(0);

		template_neuron.SetNeuronID(sum_of_all_previous_layers+j);
		template_neuron.SetCurrentValue(T(0));
		double rand_bias_weight=(rand()%1000)*0.001f-0.5f;
		template_neuron.SetBias(rand_bias_weight);
		map<int,double> wvs;
		wvs.clear();
		template_neuron.SetWeights(wvs);
		m_neurons.push_back(template_neuron);
	}
	
	// for all layers, apart from the first one, 
	// add the incoming nodes....

	int curr_index = m_num_neurons_per_layer[0]; //
	int offset_indices = 0;
	for(int i=1;i<m_num_layers;++i)
	{
		vector<int> indices_parent_nodes;
		indices_parent_nodes.clear();
		for(int j=0;j<m_num_neurons_per_layer[i-1];j++)
			indices_parent_nodes.push_back(offset_indices+j); 
		
		for(int j=0;j<m_num_neurons_per_layer[i];j++)
		{
			m_neurons[curr_index].SetIndicesParentNodes(indices_parent_nodes);
			curr_index++;
		}
	offset_indices+=m_num_neurons_per_layer[i-1];
	}
}

