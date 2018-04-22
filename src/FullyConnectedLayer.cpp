/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */
#include <iostream>
#include "FullyConnectedLayer.hpp"
#include "Mathematics.hpp"

FullyConnected_Layer::FullyConnected_Layer(int size){
	this->size=size;
	activation=NULL;
	activation_grads=NULL;
	output = NULL;
	output_grads=NULL;
	weight=NULL;
	weight_grads=NULL;
	bias=NULL;
	bias_grads=NULL;
}

FullyConnected_Layer::~FullyConnected_Layer() {
	delete output;
	delete output_grads;
	delete weight;
	delete weight_grads;
	delete bias;
	delete bias_grads;
}

/**
 * function to generate a FullyConnectedLayer.
 * this function has to be called, before starting to forward in the Layer
 * <param> activation - output of the previous layer </param>
 * <param> pre_grads - grads of the previous layer </param>
 */
bool FullyConnected_Layer::generate(Tensor *activation, Tensor *pre_grads){
	this->activation=activation;
	this->activation_grads=pre_grads;
	output = new Tensor(size,1);
	output_grads = new Tensor(size,1);
	weight = new Tensor(size*activation->getX(),activation->getY(),activation->getZ());
	int maxval = activation->getX() * activation->getY() * activation->getZ();
	for ( int i = 0; i < weight->getZ(); i++ )
			for ( int j = 0; j < weight->getY(); j++ )
				for ( int z = 0; z < weight->getX(); z++ )
					//weight->getArray(i,j)[z] = 10 * 1.0f / maxval * ((rand() / float( RAND_MAX)-0.5));
					weight->getArray(i,j)[z] = (2*(rand() / float( RAND_MAX)-0.5))/(maxval/10);
	weight_grads = new Tensor(size*activation->getX(),activation->getY(),activation->getZ());
	mathematics::set_tensor(weight_grads, 0.0);
	bias = new Tensor(size,1);
	for(int i=0; i<size;i++)bias->getArray()[i] = 0.2 * ((rand() / float( RAND_MAX)-0.5));
	//mathematics::set_tensor(bias, 0.0);
	bias_grads = new Tensor(size,1);
	mathematics::set_tensor(bias_grads, 0.0);
	return true;
}

/**
 * function to forward to the FullyConnectedLayer
 */
bool FullyConnected_Layer::forward(){

	mathematics::set_tensor(output, 0);
	//über jedes Element der Activations gehen
	#pragma omp parallel for
	for(int z_pos = 0; z_pos < activation->getZ(); z_pos++)
	{
		for(int y_pos = 0; y_pos < activation->getY(); y_pos++)
		{
			for(int x_pos = 0; x_pos < activation->getX(); x_pos++)
			{
				//für jedes Element die verschiedenen Nodes rechnen
				for(int node_index = 0; node_index < output->getX(); node_index++){
					//std::cout << pre_tensor->getArray(z_pos,y_pos)[x_pos] << " " << weight->getArray(z_pos,y_pos)[x_pos+node_index*pre_tensor->getX()] << std::endl;
					output->getArray(0,0)[node_index]+=activation->getArray(z_pos,y_pos)[x_pos]*weight->getArray(z_pos,y_pos)[x_pos+node_index*activation->getX()];
				}
			}
		}
	}
	for(int node_index = 0; node_index < output->getX(); node_index++){
		//Teilen durch Anzahl der Activations pro Node
		//output->getArray()[node_index] /= (activation->getX()*activation->getY()*activation->getZ());
		//Bias hinzufuegen
		output->getArray()[node_index] += bias->getArray()[node_index];
		//Sigmoid anwenden
		output->getArray()[node_index] = mathematics::sigmoid_once(output->getArray()[node_index]);
	}
	return true;
}

/**
 * function to backpropagate in the FullyConnectedLayer
 *
 * calculates pre_grads, gradients of the layer beyond
 * also calculates bias_grads & weight_grads of this Layer
 *
 * grads of this Layer have to be correct in order for this function to work
 */
bool FullyConnected_Layer::backward(){
	//pre_tensor[i]=weight[i+1]*node_deriv[i+1]*sigmoid_backward_derivated_once(node[i])

	//jedes Element des pre_grads
	#pragma omp parallel for
	for(int z_pos = 0; z_pos < activation_grads->getZ(); z_pos++)
	{
		for(int y_pos = 0; y_pos < activation_grads->getY(); y_pos++)
		{
			for(int x_pos = 0; x_pos < activation_grads->getX(); x_pos++)
			{
				float tmp=0;
				//jede vorherige Node hat output->getX() Gewichte die Backprppagoiert werden sollen
				for(int node_index = 0; node_index < output->getX(); node_index++){
					//tmp = (w(l)*d(l))
					tmp += weight->getArray(z_pos, y_pos)[x_pos+node_index*activation->getX()] * output_grads->getArray()[node_index];
					//grads weight von diesem Layer
					weight_grads->getArray(z_pos, y_pos)[x_pos+node_index*activation->getX()] += activation->getArray(z_pos, y_pos)[x_pos] * output_grads->getArray()[node_index];
				}

				activation_grads->getArray(z_pos,y_pos)[x_pos] = tmp * mathematics::sigmoid_backward_derivated_once(activation->getArray(z_pos,y_pos)[x_pos]);
			}
		}
	}
	//grads bias von diesem Layer
	for(int size=0; size < output->getSize(); size++){
		bias_grads->getArray()[size] += output_grads->getArray()[size];
	}

	return true;
}

/*function to fix values of weight and bias in this Layer
 *
 * <param> batch_size - </param>
 * <param> training_rate - </param>
 */
bool FullyConnected_Layer::fix(int batch_size, float training_rate){
	#pragma omp parallel for
	for(int node_index = 0; node_index < output->getX(); node_index++){
		bias->getArray()[node_index] -= training_rate/batch_size * bias_grads->getArray()[node_index];

		for(int z_pos = 0; z_pos < activation->getZ(); z_pos++){
			for(int y_pos = 0; y_pos < activation->getY(); y_pos++){
				for(int x_pos = 0; x_pos < activation->getX(); x_pos++){
					weight->getArray(z_pos,y_pos)[x_pos+node_index*activation->getX()] -= training_rate/batch_size * weight_grads->getArray(z_pos,y_pos)[x_pos+node_index*activation->getX()];

				}
			}
		}
	}

	mathematics::set_tensor(weight_grads, 0.0);
	mathematics::set_tensor(bias_grads, 0.0);
	return true;
}
