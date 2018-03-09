/*
 * Tensor.hpp
 *
 *  Created on: 06.03.2018
 *      Author: josua
 */

#ifndef TENSOR_HPP_
#define TENSOR_HPP_

class Tensor{
private:
	float *array;
	int x;
	int y;
	int z;
public:
	Tensor(int x, int y, int z);
	Tensor(int x,int y);
	int getX();
	int getY();
	int getZ();
	float *getArray(int z);
	float *getArray(int z, int y);
	float *getArray();
};

#endif /* TENSOR_HPP_ */
