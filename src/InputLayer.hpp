/*
 * InputLayer.hpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */

#ifndef INPUTLAYER_HPP_
#define INPUTLAYER_HPP_

#include "Tensor.hpp"

class Input_Layer{
private:
	int x;
	int y;

public:
	Input_Layer(int x, int y);
	virtual ~Input_Layer();

	Tensor *output;

	bool forward(float *picture);
};

#endif /* INPUTLAYER_HPP_ */
