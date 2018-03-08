/*
 * InputLayer.cpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
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
	node=new Tensor(x, y);
}

Input_Layer::~Input_Layer() {

}
Tensor *Input_Layer::getNode(){
	return node;
}
