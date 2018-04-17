/*
 * MaxPoolingLayer.hpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */

#ifndef MAXPOOLINGLAYER_HPP_
#define MAXPOOLINGLAYER_HPP_

#include "Tensor.hpp"
#include "ConvLayer.hpp"

class MaxPooling_Layer{
private:
	int x_receptive;
	int y_receptive;

public:
	MaxPooling_Layer(int x_receptive, int y_receptive);
	virtual ~MaxPooling_Layer();

	Tensor *activation;
	Tensor *output;
	Tensor *output_grads;
	Tensor *activation_grads;

	bool generate(Tensor *activation, Tensor *pre_grads);
	bool forward();
	bool backward();

};

#endif /* MAXPOOLINGLAYER_HPP_ */
