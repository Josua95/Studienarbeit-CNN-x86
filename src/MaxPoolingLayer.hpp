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
	int no_feature_maps;
	Tensor *node;
	Tensor *node_z;
	Tensor *node_deriv;
public:
	MaxPooling_Layer(int x_receptive, int y_receptive);
	virtual ~MaxPooling_Layer();

	int  getNoFeatureMaps();
	int  getXReceptive();
	int  getYReceptive();
	Tensor *getNodeTensor();
	float *getNode(int feature_map);

	bool generate(Conv_Layer *pre_layer);
	bool forward(Tensor *pre_tensor, Tensor *pre_tensor_z);
	bool backward(Tensor *post_deriv_weight, Tensor *post_deriv_bias);

};

#endif /* MAXPOOLINGLAYER_HPP_ */
