/*
 * Tensor.cpp
 *
 *  Created on: 06.03.2018
 *      Author: josua
 */

#include "Tensor.hpp"

Tensor::Tensor(int x, int y, int z){
	this->x=x;
	this->y=y;
	this->z=z;
	array = new float[z*x*y];
}

Tensor::Tensor(int x, int y){
	this->x=x;
	this->y=y;
	z=1;
	array = new float[x*y];
}

int Tensor::getX(){
	return x;
}

int Tensor::getY(){
	return y;
}

int Tensor::getZ(){
	return z;
}

float *Tensor::getArray(int z, int y){
	return array+z*this->y*x+y*x;
}

float *Tensor::getArray(int z){
	return array+z*y*x;
}
