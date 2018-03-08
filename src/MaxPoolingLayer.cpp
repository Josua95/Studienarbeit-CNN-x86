/*
 * MaxPoolingLayer.cpp
 *
 *  Created on: 29.11.2017
 *      Author: Benjamin Riedle
 */

#include "MaxPoolingLayer.hpp"
#include "ConvLayer.hpp"
#include <iostream>
#include <limits>

MaxPooling_Layer::MaxPooling_Layer(int x_receptive, int y_receptive){
	this->x_receptive = x_receptive;
	this->y_receptive = y_receptive;
	no_feature_maps = 0;
	node = NULL;

}

MaxPooling_Layer::~MaxPooling_Layer() {
	// TODO Auto-generated destructor stub
}

int MaxPooling_Layer::getNoFeatureMaps()
{
	return no_feature_maps;
}

int MaxPooling_Layer::getXReceptive()
{
	return x_receptive;
}

int MaxPooling_Layer::getYReceptive()
{
	return y_receptive;
}

float *MaxPooling_Layer::getNode(int feature_map){
	if(feature_map <= getNoFeatureMaps()){
		return node->getArray(feature_map);
	}
	std::cerr << "ERROR";
	return new float;
}

bool MaxPooling_Layer::generate(Conv_Layer *pre_layer){
	int prev_dim_x = pre_layer->getNodeTensor()->getX();
	int prev_dim_y = pre_layer->getNodeTensor()->getY();
	int prev_dim_z = pre_layer->getNodeTensor()->getZ();
	no_feature_maps = pre_layer->getNoFeatureMaps();
	int x_size = (prev_dim_x / getXReceptive());
	int y_size = (prev_dim_y / getYReceptive());

	node = new Tensor(x_size, y_size, prev_dim_z);
	return true;
}
bool MaxPooling_Layer::forward(Tensor *pre_tensor){
	int conv_x_size = pre_tensor->getX();
	int conv_y_size = pre_tensor->getY();
	int conv_z_size = pre_tensor->getZ();
	int x_step_size = getXReceptive();
	int y_step_size = getYReceptive();
	float max_node = 0.0f;
	float new_node = 0.0f;

	//Alle Feature-Maps durchgehen
	for(int feature_map = 0; feature_map < conv_z_size; feature_map++){
		//immer links oben von pooling-Flaeche
		for(int y_pos = 0; y_pos < conv_y_size; y_pos=y_pos+y_step_size){
			for(int x_pos = 0; x_pos < conv_x_size; x_pos=x_pos+x_step_size){

				//Maximalen Wert suchen
				max_node = 0.0f;
				for(int y_step = 0; y_step < y_step_size; y_step++){
					for(int x_step = 0; x_step < x_step_size; x_step++){
						new_node = pre_tensor->getArray(feature_map, y_pos+y_step)[x_pos+x_step];
						if(new_node >= max_node){
							max_node = new_node;
						}
					}
				}
				node->getArray(feature_map, y_pos/y_step_size)[x_pos/x_step_size] = max_node;
			}
		}
	}
	return true;
}
bool MaxPooling_Layer::backward(Tensor *pre_tensor){
	return true;
}
