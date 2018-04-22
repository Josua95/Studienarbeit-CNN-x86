/*
 * FullyConnectedLayer.hpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */

#ifndef FULLYCONNECTEDLAYER_HPP_
#define FULLYCONNECTEDLAYER_HPP_

#include "Tensor.hpp"

class FullyConnected_Layer{
private:
	int size;
public:
	FullyConnected_Layer(int size);
	virtual ~FullyConnected_Layer();

	Tensor *activation;
	Tensor *activation_grads;
	Tensor *output;
	Tensor *output_grads;
	Tensor *bias;
	Tensor *bias_grads;
	Tensor *weight;
	Tensor *weight_grads;

	bool generate(Tensor *activation, Tensor *pre_grads);
	bool forward();
	bool backward();
	bool fix(int batch_size, float training_rate);

};

#endif /* FULLYCONNECTEDLAYER_HPP_ */
