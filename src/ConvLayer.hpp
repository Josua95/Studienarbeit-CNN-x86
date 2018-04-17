/*
 * ConvLayer.hpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */

#ifndef CONVLAYER_HPP_
#define CONVLAYER_HPP_

#include "Tensor.hpp"

class Conv_Layer {

private:
	int x_receptive;
	int y_receptive;
	int step_size;
	int no_feature_maps;

public:
	Tensor *activation;
	Tensor *output;
	Tensor *output_grads;
	Tensor *activation_grads;
	Tensor *bias;
	Tensor *bias_grads;
	Tensor *weight;
	Tensor *weight_grads;
	Conv_Layer(int x_receptive, int y_receptive, int step_size, int no_feature_maps);
	virtual ~Conv_Layer();

	bool generate(Tensor *activation, Tensor *pre_grads);
	bool forward();
	bool backward();
	bool fix(int batch_size, float training_rate);

};

#endif /* CONVLAYER_HPP_ */
