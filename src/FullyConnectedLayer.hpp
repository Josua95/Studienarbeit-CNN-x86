/*
 * FullyConnectedLayer.hpp
 *
 *  Created on: 29.11.2017
 *      Author: Josua Benz
 */

#ifndef FULLYCONNECTEDLAYER_HPP_
#define FULLYCONNECTEDLAYER_HPP_

#include "Tensor.hpp"

class FullyConnected_Layer{
private:
	int size;
	Tensor *node;
	Tensor *node_deriv;
	Tensor *bias;
	Tensor *bias_deriv;
	Tensor *weight;
	Tensor *weight_deriv;
public:
	FullyConnected_Layer(int size);
	virtual ~FullyConnected_Layer();
	int getSize();
	Tensor *getNode();

	bool generate(Tensor *pre_node);
	bool forward(Tensor *pre_node);
	bool backward(Tensor *pre_node_deriv, Tensor *pre_node);
	bool fix(Tensor *pre_node, int batch_size, float training_rate);

};

#endif /* FULLYCONNECTEDLAYER_HPP_ */
