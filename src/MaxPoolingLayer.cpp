/*
 * MaxPoolingLayer.cpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */

#include "MaxPoolingLayer.hpp"
#include "ConvLayer.hpp"
#include "Mathematics.hpp"
#include <iostream>
#include <limits>

MaxPooling_Layer::MaxPooling_Layer(int x_receptive, int y_receptive){
	this->x_receptive = x_receptive;
	this->y_receptive = y_receptive;
	activation = NULL;
	output = NULL;
	grads = NULL;
	pre_grads = NULL;

}

MaxPooling_Layer::~MaxPooling_Layer() {
	delete output;
	delete grads;
}

bool MaxPooling_Layer::generate(Tensor *activation, Tensor *pre_grads){
	this->activation = activation;
	this->pre_grads = pre_grads;
	int x_size = (activation->getX() / x_receptive);
	int y_size = (activation->getY() / y_receptive);

	output = new Tensor(x_size, y_size, activation->getZ());
	return true;
}
bool MaxPooling_Layer::forward(){
	int conv_x_size = activation->getX();
	int conv_y_size = activation->getY();
	int conv_z_size = activation->getZ();
	int x_step_size = x_receptive;
	int y_step_size = y_receptive;
	float max_node = 0.0f;
	float new_node = 0.0f;

	//Alle Feature-Maps durchgehen
	#pragma omp for
	for(int feature_map = 0; feature_map < conv_z_size; feature_map++){
		//immer links oben von pooling-Flaeche
		for(int y_pos = 0; y_pos < conv_y_size; y_pos=y_pos+y_step_size){
			for(int x_pos = 0; x_pos < conv_x_size; x_pos=x_pos+x_step_size){

				//Maximalen Wert suchen
				max_node = -std::numeric_limits<float>::max();
				for(int y_step = 0; y_step < y_step_size; y_step++){
					for(int x_step = 0; x_step < x_step_size; x_step++){
						new_node = activation->getArray(feature_map, y_pos+y_step)[x_pos+x_step];
						if(new_node >= max_node){
							max_node = new_node;
						}
					}
				}
				output->getArray(feature_map, y_pos/y_step_size)[x_pos/x_step_size] = max_node;
			}
		}
	}
	return true;
}
bool MaxPooling_Layer::backward(){

	int step = activation->getX() - output->getX() + 1;

	//Ueber alle Feature Maps des vorherigen Layers gehen
	for (int z_pos = 0; z_pos < activation->getZ(); z_pos++){
		//Ueber Anfangselement von jedem Pooling-Rechteck iterieren
		for (int y_pos = 0; y_pos < activation->getY()-step; y_pos = y_pos++){
			for (int x_pos = 0; x_pos < activation->getX()-step; x_pos = x_pos++){

				int max_node_index_x = 0;
				int max_node_index_y = 0;
				float max_node_value = - std::numeric_limits<float>::max();

				//benachbarte Elemente durchgehen und max ermitteln
				for (int l = 0; l < step; l++){
					for (int m = 0; m < step; m++){
						//pre_grads auf 0 setzten
						pre_grads->getArray(z_pos, y_pos+l)[x_pos+m];

						if (max_node_value < activation->getArray(z_pos,y_pos+l)[x_pos+m]){
							max_node_value = activation->getArray(z_pos,y_pos+l)[x_pos+m];
							max_node_index_x = m;
							max_node_index_y = l;
						}
					}
				}
				//Wert Zurueckfuehren an den Node mit dem größten Wert
				pre_grads->getArray(z_pos,y_pos+max_node_index_y)[x_pos+max_node_index_x]= grads->getArray(z_pos,y_pos)[x_pos];
			}
		}
	}
	return true;
}
