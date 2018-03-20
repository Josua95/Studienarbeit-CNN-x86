/*
 * mathematics.cpp
 *
 *  Created on: 05.12.2017
 *      Author: Florian
 */

#include "Mathematics.hpp"

#include "math.h"
#include <iostream>

namespace mathematics {


float sigmoid_once(float in)
{
	double temp = exp(-in);
	return (float)(1 / (1+temp));
}

float sigmoid_backward_derivated_once(float activation)
{
	return activation * (1 - activation);
}

float sigmoid_once_inverse(float a){
	return -log(1/a -1);
}

void sigmoid(float *in, float *out, int size)
{
	for(; size>0; size--, in++, out++)
	{
		*out = sigmoid_once(*in);
	}
}

void sigmoid_backward_derivated(float *activation, float *derivatives, int size)
{
	for(; size>0; size--, activation++, derivatives++)
	{
		*derivatives = sigmoid_backward_derivated_once(*activation);
	}
}


void softmax(float *in, float *out, int size)
{
	double sum=0;
	for(int i=0; i<size; i++)
	{
		sum += exp(in[i]);
	}
	for(int i=0; i<size; i++)
	{
		out[i] = (float)(exp(in[i]) / sum);
	}
}


float cross_entropy(float *calculated, float *expected, int size)
{
	double sum=0;
	for(; size>0; size--, expected++, calculated++)
	{
		sum += - (*expected) * log(*calculated);
	}
	return (float)sum;
}

float get_cost(float *output, float *labels, int size)
{
	float *normalized;
	float ret;
	normalized = new float[size];
	softmax(output, normalized, size);
	ret = cross_entropy(normalized, labels, size);
	delete[] normalized;
	return ret;
}

void get_cost_derivatives(float *output, float *labels, float *derivatives, int size)
{
	for(; size>0; size--, output++, labels++, derivatives++)
	{
		*derivatives = *output - *labels;
	}
}

void set_tensor(Tensor *tensor, float value){
	#pragma omp for
	for(int z_pos=0; z_pos < tensor->getZ(); z_pos++){
		for(int y_pos = 0; y_pos < tensor->getY(); y_pos++){
			for(int x_pos = 0; x_pos < tensor->getX();x_pos++){
				tensor->getArray(z_pos, y_pos)[x_pos]= value;
			}
		}
	}
}

void set_tensor_random(Tensor *tensor){
	#pragma omp for
	for(int z_pos=0; z_pos < tensor->getZ(); z_pos++){
		for(int y_pos = 0; y_pos < tensor->getY(); y_pos++){
			for(int x_pos = 0; x_pos < tensor->getX();x_pos++){
				float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				tensor->getArray(z_pos, y_pos)[x_pos]= r;
			}
		}
	}
}

void printTensor(Tensor *tensor){
	for(int z_pos=0; z_pos<tensor->getZ(); z_pos++){
		for(int y_pos=0; y_pos<tensor->getY(); y_pos++){
			for(int x_pos=0; x_pos<tensor->getX(); x_pos++){
				std::cout << tensor->getArray(z_pos, y_pos)[x_pos] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl << std::endl;
	}
}

} /* namespace mathematics */
