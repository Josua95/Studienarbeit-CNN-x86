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

bool FullyConnected_Layer::generate(Tensor *pre_node){
	node = new Tensor(size,1);
	node_deriv = new Tensor(size,1);
	weight = new Tensor(size*pre_node->getX(),pre_node->getY(),pre_node->getZ());
	mathematics::set_tensor_random(weight);
	weight_deriv = new Tensor(size*pre_node->getX(),pre_node->getY(),pre_node->getZ());
	mathematics::set_tensor_random(weight_deriv);
	bias = new Tensor(size,1);
	mathematics::set_tensor_random(bias);
	bias_deriv = new Tensor(size,1);
	mathematics::set_tensor_random(bias_deriv);
	return true;
}
bool FullyConnected_Layer::forward(Tensor *pre_node){

	mathematics::set_tensor(node, 0);
	#pragma omp for
	for(int z_pos = 0; z_pos < pre_node->getZ(); z_pos++)
	{
		for(int y_pos = 0; y_pos < pre_node->getY(); y_pos++)
		{
			for(int x_pos = 0; x_pos < pre_node->getX(); x_pos++)
			{
				for(int node_index = 0; node_index < node->getX(); node_index++){
					//std::cout << pre_tensor->getArray(z_pos,y_pos)[x_pos] << " " << weight->getArray(z_pos,y_pos)[x_pos+node_index*pre_tensor->getX()] << std::endl;
					node->getArray(0,0)[node_index]+=pre_node->getArray(z_pos,y_pos)[x_pos]*weight->getArray(z_pos,y_pos)[x_pos+node_index*pre_node->getX()];
				}
			}
		}
	}
	for(int node_index = 0; node_index < node->getX(); node_index++){
		node->getArray()[node_index] /= (pre_node->getX()*pre_node->getY()*pre_node->getZ());
		node->getArray()[node_index] *= bias->getArray()[node_index];
		node->getArray()[node_index] = mathematics::sigmoid_once(node->getArray()[node_index]);
	}
	return true;
}


bool FullyConnected_Layer::backward(Tensor *pre_node_deriv, Tensor *pre_node){
	//TODO Backpropagation FullyConnectedLayer
	//pre_tensor[i]=weight[i+1]*node_deriv[i+1]*sigmoid_backward_derivated_once(node[i])
	#pragma omp for
	//jedes Element des vorherigen node_deriv
	for(int z_pos = 0; z_pos < pre_node_deriv->getZ(); z_pos++)
	{
		for(int y_pos = 0; y_pos < pre_node_deriv->getY(); y_pos++)
		{
			for(int x_pos = 0; x_pos < pre_node_deriv->getX(); x_pos++)
			{
				float tmp=0;
				//jedes Element dieser Node_list
				for(int node_index = 0; node_index < node->getX(); node_index++){
					//tmp = (w(l)*d(l))
					tmp += weight->getArray(z_pos, y_pos)[x_pos+node_index*pre_node->getX()]*node_deriv->getArray()[node_index];
				}
				//d(l-1) += tmp*sigmoid'(node(l-1))
				pre_node_deriv->getArray(z_pos,y_pos)[x_pos] = tmp * mathematics::sigmoid_backward_derivated_once(pre_node->getArray(z_pos,y_pos)[x_pos]);
			}
		}
	}

	return true;
}

bool FullyConnected_Layer::fix(Tensor *pre_node, int batch_size, float training_rate){
	#pragma omp for
	for(int node_index = 0; node_index < node->getX(); node_index++){
		//TODO Gradient Descent
		bias->getArray()[node_index] -= training_rate/batch_size * node_deriv->getArray()[node_index];

		for(int z_pos = 0; z_pos < pre_node->getZ(); z_pos++)
		{
			for(int y_pos = 0; y_pos < pre_node->getY(); y_pos++)
			{
				for(int x_pos = 0; x_pos < pre_node->getX(); x_pos++)
				{
					//TODO Gradient Descent
					weight->getArray(z_pos,y_pos)[x_pos+node_index*pre_node] -= training_rate/batch_size * pre_node(z_pos,y_pos)[x_pos] * node_deriv->getArray(node_index);

				}
			}
		}
	}
	return true;
}
