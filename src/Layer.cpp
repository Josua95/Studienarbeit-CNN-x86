/*
 * Layer.cpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */


#include "Layer.hpp"

/*
 * The default constructor for class Layer sets its size
 *
 * <param> LAYER_TYPE layer_type </param>
 *
 */
Layer::Layer(LAYER_TYPE layer_type){
	type = layer_type;
}

Layer::~Layer()
{

}

LAYER_TYPE Layer::getLayerType()
{
	return type;
}
