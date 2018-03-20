/*
 * FullyConnectedLayer.hpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#ifndef FULLYCONNECTEDLAYER_HPP_
#define FULLYCONNECTEDLAYER_HPP_

#include "Tensor.hpp"

class FullyConnected_Layer{
private:
	int size;
	Tensor *node;
	Tensor *node_z;
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

	bool generate(Tensor *pre_tensor);
	bool forward(Tensor *pre_tensor);
	bool backward(Tensor *post_deriv_weight, Tensor *post_deriv_bias);

};

#endif /* FULLYCONNECTEDLAYER_HPP_ */
