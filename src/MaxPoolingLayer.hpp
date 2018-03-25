/*
 * MaxPoolingLayer.hpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
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
	Tensor *pre_grads;

	bool generate(Tensor *activation, Tensor *pre_grads);
	bool forward();
	bool backward(Tensor *post_grads);

};

#endif /* MAXPOOLINGLAYER_HPP_ */
