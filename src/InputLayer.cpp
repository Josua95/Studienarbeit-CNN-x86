/*
 * InputLayer.cpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */

#include "InputLayer.hpp"

/*
 * The default constructor of Conv_Layer uses the
 * constructor of the base class Layer to set number
 * of layers nodes
 *
 * <params> int size - number of layers nodes </params>
 *
 */
Input_Layer::Input_Layer(int x, int y){
	this->x=x;
	this->y=y;
	output=new Tensor(x, y);
}

Input_Layer::~Input_Layer() {

}
Tensor *Input_Layer::getNode(){
	return output;
}

bool Input_Layer::forward(float *picture){
	#pragma omp for
	for(int pos_y=0; pos_y<y; pos_y++){
		for(int pos_x=0;pos_x<x;pos_x++){
			output->getArray(0,pos_y)[pos_x]=picture[pos_y*x+pos_x];
		}
	}
	return true;
}
