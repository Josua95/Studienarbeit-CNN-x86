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
	output_grads = NULL;
	activation_grads = NULL;

}

MaxPooling_Layer::~MaxPooling_Layer() {
	delete output;
	delete output_grads;
}

bool MaxPooling_Layer::generate(Tensor *activation, Tensor *pre_grads){
	this->activation = activation;
	this->activation_grads = pre_grads;
	int x_size = (activation->getX() / x_receptive);
	int y_size = (activation->getY() / y_receptive);

	output = new Tensor(x_size, y_size, activation->getZ());
	output_grads = new Tensor(x_size, y_size, activation->getZ());
	return true;
}
bool MaxPooling_Layer::forward(){

	//Alle Feature-Maps durchgehen
	#pragma omp parallel for
	for(int feature_map = 0; feature_map < activation->getZ(); feature_map++){
		//immer links oben von pooling-Flaeche
		for(int y_pos = 0; y_pos < activation->getY(); y_pos+=y_receptive){
			for(int x_pos = 0; x_pos < activation->getX(); x_pos+=x_receptive){

				float max_node = 0.0f;
				float new_node = 0.0f;
				//Maximalen Wert suchen
				max_node = 0;
				for(int y_step = 0; y_step < y_receptive; y_step++){
					for(int x_step = 0; x_step < x_receptive; x_step++){
						new_node = activation->getArray(feature_map, y_pos+y_step)[x_pos+x_step];
						if(new_node >= max_node){
							max_node = new_node;
						}
					}
				}
				output->getArray(feature_map, y_pos/y_receptive)[x_pos/x_receptive] = max_node;
			}
		}
	}
	return true;
}
bool MaxPooling_Layer::backward(){

	//Ueber alle Feature Maps des vorherigen Layers gehen
	#pragma omp parallel for
	for (int z_pos = 0; z_pos < activation->getZ(); z_pos++){
		//Ueber Anfangselement von jedem Pooling-Rechteck iterieren
		for (int y_pos = 0; y_pos < activation->getY(); y_pos+= y_receptive){
			for (int x_pos = 0; x_pos < activation->getX(); x_pos+=x_receptive){

				int max_node_index_x = 0;
				int max_node_index_y = 0;
				float max_node_value = 0;

				//benachbarte Elemente durchgehen und max ermitteln
				for (int y_step = 0; y_step < y_receptive; y_step++){
					for (int x_step = 0; x_step < x_receptive; x_step++){
						//pre_grads auf 0 setzten
						activation_grads->getArray(z_pos, y_pos+y_step)[x_pos+x_step] = 0;

						if (max_node_value < activation->getArray(z_pos,y_pos+y_step)[x_pos+x_step]){
							max_node_value = activation->getArray(z_pos,y_pos+y_step)[x_pos+x_step];
							max_node_index_x = x_step;
							max_node_index_y = y_step;
						}
					}
				}
				//Wert Zurueckfuehren an den Node mit dem größten Wert
				activation_grads->getArray(z_pos,y_pos+max_node_index_y)[x_pos+max_node_index_x]= output_grads->getArray(z_pos,y_pos/y_receptive)[x_pos/x_receptive];
			}
		}
	}
	return true;
}
