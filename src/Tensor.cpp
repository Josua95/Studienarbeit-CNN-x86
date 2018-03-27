/*
 * Tensor.cpp
 *
 *  Created on: 06.03.2018
 *      Author: Josua Benz
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

Tensor::~Tensor(){
	delete[] array;
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

int Tensor::getSize(){
	return x*y*z;
}

float *Tensor::getArray(int z, int y){
	return array+z*this->y*x+y*x;
}

float *Tensor::getArray(int z){
	return array+z*y*x;
}

float *Tensor::getArray(){
	return array;
}
