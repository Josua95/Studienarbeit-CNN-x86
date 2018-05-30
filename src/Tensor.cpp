/*
 * Tensor.cpp
 *
 *  Created on: 06.03.2018
 *      Author: Josua Benz
 */

#include "Tensor.hpp"
#include <assert.h>

/**
 * Constructor mit Angabe von allen 3 Dimensionen des Tensors
 * x: Dimension x-Richtung
 * y: Dimension y-Richtung
 * z: Dimension z-Richtung
 */
Tensor::Tensor(int x, int y, int z){
	assert(x>0);
	assert(y>0);
	assert(z>0);
	this->x=x;
	this->y=y;
	this->z=z;
	array = new float[z*x*y];
}

/**
 * Konstruktor mit Angabe von 2 Dimensionen (Matrix), Dimension z wird auf 1 gesetzt
 * x: Dimension x-Richtung
 * y: Dimension y-Richtung
 */
Tensor::Tensor(int x, int y){
	assert(x>0);
	assert(y>0);
	this->x=x;
	this->y=y;
	z=1;
	array = new float[x*y];
}

Tensor::~Tensor(){
	delete[] array;
}

/**
 * return: Größe in x-Richtung
 */
int Tensor::getX(){
	return x;
}

/**
 * return: Größe in y-Richtung
 */
int Tensor::getY(){
	return y;
}


/**
 * return: Größe in z-Richtung
 */
int Tensor::getZ(){
	return z;
}

/**
 * return: Anzahl an Elementen des Tensors (x*y*z)
 */
int Tensor::getSize(){
	return x*y*z;
}

/**
 * return: Zeiger auf Anfang einer Zeile
 * z: Seitennummer
 * y: Zeilennummer
 */
float *Tensor::getArray(int z, int y){
	assert(z < this->z);
	assert(y < this->y);
	return array + z * this->y * x + y * x;
}

/**
 * return: Zeiger auf Anfang einer Seite
 * z: Seitennummer
 */
float *Tensor::getArray(int z){
	assert(z < this->z);
	return array + z * y * x;
}

/**
 * return: Zeiger auf Anfang des Tensors
 */
float *Tensor::getArray(){
	return array;
}
