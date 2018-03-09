/**
 * ConvLayer.cpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */

#include "ConvLayer.hpp"
#include <iostream>

/**
 * The constructor of a convolutional layer needs the specification of
 * the local receptive fields and the step size to generate an output
 *
 * <param> int x_receptive - size of receptive field in x-direction </param>
 * <param> int y_receptive - size of receptive field in y-direction </param>
 * <param> int step_size - step size to move receptive field </param>
 */
Conv_Layer::Conv_Layer(int x_receptive, int y_receptive, int step_size, int no_feature_maps){
	this->step_size = step_size;
	this->x_receptive = x_receptive;
	this->y_receptive = y_receptive;
	this->no_feature_maps = no_feature_maps;
	weight=NULL;
	bias=NULL;
	node=NULL;
}

Conv_Layer::~Conv_Layer() {
}

int Conv_Layer::getNoFeatureMaps()
{
	return no_feature_maps;
}

int Conv_Layer::getXReceptive()
{
	return x_receptive;
}

int Conv_Layer::getYReceptive()
{
	return y_receptive;
}

int Conv_Layer::getStepSize()
{
	return step_size;
}

Tensor *Conv_Layer::getNodeTensor(){
	return node;
}

float *Conv_Layer::getBias(int feature_map){
	return bias->getArray(feature_map);
}

float *Conv_Layer::getNode(int feature_map){
	return node->getArray(feature_map);
}

float *Conv_Layer::getWeight(int feature_map){
	return weight->getArray(feature_map);
}
bool Conv_Layer::generate(Tensor *pre_tensor){

	int pre_x_size = pre_tensor->getX();
	int pre_y_size = pre_tensor->getY();
	int pre_z_size = pre_tensor->getZ();

	int x_size = (pre_x_size - getXReceptive() + 1) / getStepSize();
	int y_size = (pre_y_size - getYReceptive() + 1) / getStepSize();

	weight = new Tensor(getXReceptive(), getYReceptive(), getNoFeatureMaps());
	bias = new Tensor(getXReceptive(), getYReceptive(), getNoFeatureMaps());
	//TODO set all equal
	node = new Tensor(x_size, y_size, pre_z_size*getNoFeatureMaps());
	return true;
}

bool Conv_Layer::forward(Tensor *pre_tensor){
	int pre_x_size = pre_tensor->getX();
	int pre_y_size = pre_tensor->getY();
	int pre_z_size = pre_tensor->getZ();

	//Alle Feature_maps des vorherigen Layers
	#pragma omp parallel for
	for(int pre_feature=0; pre_feature < pre_z_size; pre_feature++){
		//jedes Element der Matrix von Input Layer durchlaufen
		for(int y_pos = 0; y_pos < pre_y_size; y_pos++){
			for(int x_pos = 0; x_pos < pre_x_size;x_pos++){

				//Ausrechnen, welche Gewichte genommen werden können
				int start_x_rec=0;
				int stop_x_rec=x_receptive-1;
				if(x_pos < x_receptive-1) stop_x_rec = x_pos;
				else if(x_pos > pre_x_size-x_receptive) start_x_rec = x_receptive + x_pos - pre_x_size;

				int start_y_rec=0;
				int stop_y_rec=y_receptive-1;
				if(y_pos < y_receptive-1) stop_y_rec = y_pos;
				else if(y_pos > pre_y_size-y_receptive) start_y_rec = y_receptive + y_pos - pre_y_size;

				//Ueber Gewichte iterieren und in float-Array speichern
				for(int feature_map=0;feature_map<getNoFeatureMaps();feature_map++){
					//Ueber Gewichte iterieren und in float-Array speichern
					for(int y_rec = start_y_rec; y_rec <= stop_y_rec ; y_rec++){
						for(int x_rec = start_x_rec; x_rec <= stop_x_rec ; x_rec++){
							node->getArray(feature_map, y_pos-y_rec)[x_pos-x_rec] += pre_tensor->getArray(pre_feature,y_pos)[x_pos]*getWeight(feature_map)[y_rec*y_receptive+x_rec];
						}
					}
				}
			}
		}
	}
	//Bias hizufuegen
	for(int pre_feature=0; pre_feature < pre_z_size; pre_feature++){
		for(int y_pos = 0; y_pos < node->getY(); y_pos++){
			for(int x_pos = 0; x_pos < node->getX();x_pos++){
				for(int feature_map=0;feature_map<getNoFeatureMaps();feature_map++){
					node->getArray(feature_map*pre_feature, y_pos)[+x_pos] += getBias(feature_map)[y_pos*node->getY()+x_pos];
				}
			}
		}
	}
	return true;
}

bool Conv_Layer::backward(Tensor *pre_tensor){
	return true;
}
