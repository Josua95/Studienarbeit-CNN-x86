/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */
#include <iostream>
#include "FullyConnectedLayer.hpp"
#include "Mathematics.hpp"

FullyConnected_Layer::FullyConnected_Layer(int size){
	this->size=size;
	node = NULL;
	node_deriv=NULL;
	weight=NULL;
	weight_deriv=NULL;
	bias=NULL;
	bias_deriv=NULL;
}

FullyConnected_Layer::~FullyConnected_Layer() {
	// TODO Auto-generated destructor stub
}

int FullyConnected_Layer::getSize(){
	return size;
}

Tensor *FullyConnected_Layer::getNode(){
	return node;
}

bool FullyConnected_Layer::generate(Tensor *pre_tensor){
	node = new Tensor(size,1);
	node_deriv = new Tensor(size,1);
	weight = new Tensor(size*pre_tensor->getX(),pre_tensor->getY(),pre_tensor->getZ());
	for(int z_pos=0; z_pos < weight->getZ(); z_pos++){
		for(int y_pos = 0; y_pos < weight->getY(); y_pos++){
			for(int x_pos = 0; x_pos < weight->getX();x_pos++){
				weight->getArray(z_pos, y_pos)[x_pos]= 0.5;
			}
		}
	}
	weight_deriv = new Tensor(size*pre_tensor->getX(),pre_tensor->getY(),pre_tensor->getZ());
	mathematics::set_tensor(weight_deriv, 0.5);
	bias = new Tensor(size*pre_tensor->getX(),pre_tensor->getY(),pre_tensor->getZ());
	mathematics::set_tensor(bias, 0.5);
	bias_deriv = new Tensor(size*pre_tensor->getX(),pre_tensor->getY(),pre_tensor->getZ());
	mathematics::set_tensor(bias_deriv, 0.5);
	return true;
}
bool FullyConnected_Layer::forward(Tensor *pre_tensor){

	mathematics::set_tensor(node, 0);
	//#pragma omp parallel for
	for(int z_pos = 0; z_pos < pre_tensor->getZ(); z_pos++)
	{
		for(int y_pos = 0; y_pos < pre_tensor->getY(); y_pos++)
		{
			for(int x_pos = 0; x_pos < pre_tensor->getX(); x_pos++)
			{
				for(int node_index = 0; node_index < node->getX(); node_index++){
					//std::cout << pre_tensor->getArray(z_pos,y_pos)[x_pos] << " " << weight->getArray(z_pos,y_pos)[x_pos+node_index*pre_tensor->getX()] << std::endl;
					node->getArray(0,0)[node_index]+=pre_tensor->getArray(z_pos,y_pos)[x_pos]*weight->getArray(z_pos,y_pos)[x_pos+node_index*pre_tensor->getX()]+bias->getArray(z_pos,y_pos)[x_pos+node_index*pre_tensor->getX()];
				}
			}
		}
	}
	for(int node_index = 0; node_index < node->getX(); node_index++){
		node->getArray(0,0)[node_index] = node->getArray(0,0)[node_index]/(pre_tensor->getX()*pre_tensor->getY()*pre_tensor->getZ());
		node->getArray(0,0)[node_index]=mathematics::sigmoid_once(node->getArray(0,0)[node_index]);
	}
	return true;
}
bool FullyConnected_Layer::backward(Tensor *pre_tensor){

	return true;
}
