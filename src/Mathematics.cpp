/*
 * mathematics.cpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */

#include "Mathematics.hpp"

#include "math.h"
#include <stdlib.h>
#include <iostream>

namespace mathematics {

/**
 * Berechnet die Sigmoid-Funktion aus dem Input
 * in: Input der zur Berechnung der Sigmoid-Funktion verwendet werden soll
 */
float sigmoid_forward(float in)
{
	double temp = exp(-in);
	return (float)(1 / (1+temp));
}

/**
 * Berechnet die inversen Sigmoid-Funktion aus dem Input
 * in: Input der zur Berechnung der inversern Sigmoid-Funktion verwendet werden soll
 */
float sigmoid_backward(float activation)
{
	return activation * (1 - activation);
}

/**
 * Berechnung der Softmax-Werte aus dem Array an Inputs
 * in: Zeiger auf Float-Array mit den Inputs
 * out: Zeiger auf die Adresse, an der die Outputs geschrieben werden sollen
 * size: Länder des Arrays Inputs
 */
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

/**
 * Setzt einen ganzen Tensor auf einen Wert
 * tensor: Zeiger auf Tensor
 * value: Wert, der jedes Element des Tensors bekommen soll
 */
void set_tensor(Tensor *tensor, float value){
	#pragma omp parallel for
	for(int z_pos=0; z_pos < tensor->getZ(); z_pos++){
		for(int y_pos = 0; y_pos < tensor->getY(); y_pos++){
			for(int x_pos = 0; x_pos < tensor->getX();x_pos++){
				tensor->getArray(z_pos, y_pos)[x_pos]= value;
			}
		}
	}
}

/**
 * Setzt einen ganzen Tensor auf einen zufälligen Wert
 * tensor: Zeiger auf Tensor
 */
void set_tensor_random(Tensor *tensor){
	#pragma omp parallel for
	for(int z_pos=0; z_pos < tensor->getZ(); z_pos++){
		for(int y_pos = 0; y_pos < tensor->getY(); y_pos++){
			for(int x_pos = 0; x_pos < tensor->getX();x_pos++){
				float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				r/=100;
				tensor->getArray(z_pos, y_pos)[x_pos]= r;
			}
		}
	}
}

/**
 * Ausgabe eines Tensors in der Konsole
 * tensor: Zeiger auf Tensor
 */
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
