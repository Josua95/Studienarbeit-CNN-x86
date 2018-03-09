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
	Tensor *bias;
	Tensor *weight;
public:
	FullyConnected_Layer(int size);
	virtual ~FullyConnected_Layer();
	int getSize();

	bool generate(Tensor *pre_tensor);
	bool forward(Tensor *pre_tensor);
	bool backward(Tensor *pre_tensor);

};

#endif /* FULLYCONNECTEDLAYER_HPP_ */
