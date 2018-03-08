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
	Tensor *node;
	Tensor *bias;
	Tensor *weight;
public:
	Conv_Layer(int x_receptive, int y_receptive, int step_size, int no_feature_maps);
	virtual ~Conv_Layer();
	int  getXReceptive();
	int  getYReceptive();
	int  getStepSize();
	int  getNoFeatureMaps();

	Tensor *getNodeTensor();
	float *getNode(int feature_map);
	float *getBias(int feature_map);
	float *getWeight(int feature_map);

	bool generate(Tensor *pre_tensor);
	bool forward(Tensor *pre_tensor);
	bool backward(Tensor *pre_tensor);

};

#endif /* CONVLAYER_HPP_ */
