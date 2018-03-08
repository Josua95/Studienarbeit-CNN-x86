/*
 * Layer.hpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */

#ifndef LAYER_HPP_
#define LAYER_HPP_

#include <vector>

typedef enum {
	INPUT_LAYER, CONV_LAYER, POOLING_LAYER, FULLY_CONNECTED_LAYER,
	DROPOUT_LAYER
} LAYER_TYPE;

using namespace std;

class Layer
{
private:
	LAYER_TYPE type; /* type of this layer */

public:
	Layer(LAYER_TYPE layer_type);
	virtual ~Layer();

	LAYER_TYPE getLayerType();

	virtual bool generate(Layer *pre_layer);
	virtual bool forward(Layer *pre_layer);
	virtual bool backward(Layer *pre_layer);
};

#endif /* LAYER_HPP_ */
