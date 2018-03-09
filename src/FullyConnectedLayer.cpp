/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#include "FullyConnectedLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "ConvLayer.hpp"
#include "InputLayer.hpp"

FullyConnected_Layer::FullyConnected_Layer(int size){
	this->size=size;
	node = new Tensor(size,1);
	weight=0;
	bias=0;
}

FullyConnected_Layer::~FullyConnected_Layer() {
	// TODO Auto-generated destructor stub
}

int FullyConnected_Layer::getSize(){
	return size;
}

bool FullyConnected_Layer::generate(Tensor *pre_tensor){
	int weight_size = pre_tensor->getX()*pre_tensor->getY()*pre_tensor->getZ();
	node = new Tensor(size,1);
	weight = new Tensor(size, weight_size);
	bias = new Tensor(size, weight_size);
	return true;
}
bool FullyConnected_Layer::forward(Tensor *pre_tensor){

	#pragma omp parallel for
	for(int z_pos = 0; z_pos < pre_tensor->getZ(); z_pos++)
	{
		for(int y_pos = 0; y_pos < pre_tensor->getY(); y_pos++)
		{
			for(int x_pos = 0; x_pos < pre_tensor->getX(); x_pos++)
			{
				for(int node_index = 0; node_index < node->getX(); node_index++){
					node->getArray(1,1)[node_index]+=pre_tensor->getArray(z_pos,y_pos)[x_pos]*weight->getArray(0,x_pos*y_pos*z_pos)[node_index]+bias->getArray(0,x_pos*y_pos*z_pos)[node_index];
				}
			}
		}
	}
	return true;
}
bool FullyConnected_Layer::backward(Tensor *pre_tensor){
	return true;
}
