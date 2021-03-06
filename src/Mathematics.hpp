/*
 * mathematics.h
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */

#ifndef MATHEMATICS_HPP_
#define MATHEMATICS_HPP_

#include "Tensor.hpp"

namespace mathematics {

float sigmoid_forward(float in);
float sigmoid_backward(float activation);

void sigmoid(float *in, float *out, int size);
void sigmoid_backward_derivated(float *activation, float *derivatives, int size);


void softmax(float *in, float *out, int size);
float cross_entropy(float *calculated, float *expected, int size);

float get_cost(float *output, float *labels, int size);
void get_cost_derivatives(float *output, float *labels, float *derivatives, int size);

void set_tensor(Tensor *tensor, float value);
void set_tensor_random(Tensor *tensor);
void printTensor(Tensor *tensor);
}


#endif /* MATHEMATICS_HPP_ */
