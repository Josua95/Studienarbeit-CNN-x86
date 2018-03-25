/*
 * Tensor.hpp
 *
 *  Created on: 06.03.2018
 *      Author: josua
 */

#ifndef TENSOR_HPP_
#define TENSOR_HPP_

struct Tensor{
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
	Tensor operator()(int x, int y, int z){
		return this->getArray(z,y)[x];
	}
};


#endif /* TENSOR_HPP_ */
