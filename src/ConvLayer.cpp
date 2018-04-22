/**
 * ConvLayer.cpp
 *
 *  Created on: 01.03.2018
 *      Author: Josua Benz
 */

#include "ConvLayer.hpp"
#include "Mathematics.hpp"
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
	weight_grads=NULL;
	bias=NULL;
	bias_grads=NULL;
	output=NULL;
	output_grads=NULL;
	activation_grads=NULL;
	activation=NULL;
}

Conv_Layer::~Conv_Layer() {
}

bool Conv_Layer::generate(Tensor *activation, Tensor *pre_grads){

	int x_size = (activation->getX() - x_receptive + 1) / step_size;
	int y_size = (activation->getY() - y_receptive + 1) / step_size;

	this->activation_grads=pre_grads;
	this->activation=activation;

	weight = new Tensor(x_receptive, y_receptive, no_feature_maps*activation->getZ());
	//mathematics::set_tensor_random(weight);
	//mathematics::set_tensor(weight, 0.0);
	int maxval = weight->getX() * weight->getY() * activation->getZ();
	for ( int i = 0; i < weight->getZ(); i++ )
		for ( int j = 0; j < weight->getY(); j++ )
			for ( int z = 0; z < weight->getX(); z++ )
				//Wert im Bereich +-0.5
				//weight->getArray(i,j)[z] = 10.0 * 1.0f / maxval * ((rand() / float( RAND_MAX)-0.5));
				weight->getArray(i,j)[z] = (2*(rand() / float( RAND_MAX)-0.5))/(maxval/10);
	weight_grads = new Tensor(x_receptive, y_receptive, no_feature_maps*activation->getZ());
	mathematics::set_tensor(weight_grads, 0.0);
	bias = new Tensor(1,1, no_feature_maps);
	for(int i=0; i< no_feature_maps; i++)bias->getArray()[i] = 0.2 * ((rand() / float( RAND_MAX)-0.5));

	//mathematics::set_tensor(bias, 0.0);
	bias_grads = new Tensor(1, 1, no_feature_maps);
	mathematics::set_tensor(bias_grads, 0.0);
	output = new Tensor(x_size, y_size, no_feature_maps);
	output_grads = new Tensor(x_size, y_size, no_feature_maps);
	return true;
}

bool Conv_Layer::forward(){
	//Alle Nodes auf 0 setzen
	mathematics::set_tensor(output, 0.0);

	//Alle Elemente des vorherigen Layers
	#pragma omp parallel for
	for(int pre_z_pos=0; pre_z_pos < activation->getZ(); pre_z_pos++){
		//jedes Element der Matrix von Input Layer durchlaufen
		for(int pre_y_pos = 0; pre_y_pos < activation->getY(); pre_y_pos++){
			for(int pre_x_pos = 0; pre_x_pos < activation->getX();pre_x_pos++){

				//Ausrechnen, welche Gewichte genommen werden können
				int start_x_rec=0;
				int stop_x_rec=x_receptive-1;
				if(pre_x_pos < x_receptive-1) stop_x_rec = pre_x_pos;
				else if(pre_x_pos > activation->getX()-x_receptive) start_x_rec = x_receptive + pre_x_pos - activation->getX();

				int start_y_rec=0;
				int stop_y_rec=y_receptive-1;
				if(pre_y_pos < y_receptive-1) stop_y_rec = pre_y_pos;
				else if(pre_y_pos > activation->getY()-y_receptive) start_y_rec = y_receptive + pre_y_pos - activation->getY();

				//über verschiedene Features
				for(int z_pos=pre_z_pos;z_pos < weight->getZ();z_pos+=activation->getZ()){
					//Ueber Gewichte iterieren und in float-Array speichern
					for(int y_rec = start_y_rec; y_rec <= stop_y_rec ; y_rec++){
						for(int x_rec = start_x_rec; x_rec <= stop_x_rec ; x_rec++){
							output->getArray(z_pos/activation->getZ(), pre_y_pos-y_rec)[pre_x_pos-x_rec] += activation->getArray(pre_z_pos,pre_y_pos)[pre_x_pos] * weight->getArray(z_pos,y_rec)[x_rec];
						}
					}
				}
			}
		}
	}
	//Bias hizufuegen & Sigmoid
	#pragma omp parallel for
	for(int y_pos = 0; y_pos < output->getY(); y_pos++){
		for(int x_pos = 0; x_pos < output->getX();x_pos++){
			for(int feature_map=0;feature_map<no_feature_maps;feature_map++){
				//Bias hinzufuegen
				output->getArray(feature_map, y_pos)[x_pos] += bias->getArray(feature_map, 0)[0];
				//Sigmoid anwenden
				output->getArray(feature_map, y_pos)[x_pos] = mathematics::sigmoid_once(output->getArray(feature_map, y_pos)[x_pos]);
			}
		}
	}
	return true;
	/*for ( int filter = 0; filter < no_feature_maps; filter++ ){

		for ( int x = 0; x < output->getX(); x++ ){
			for ( int y = 0; y < output->getY(); y++ ){
				float sum = 0;
				for ( int i = 0; i < weight->getX(); i++ )
					for ( int j = 0; j < weight->getY(); j++ )
						for ( int z = 0; z < activation->getZ(); z++ )
							sum += weight->getArray(filter*activation->getZ()+z, j)[i]*activation->getArray(z, y+i)[x+i];

				output->getArray(filter, y)[x] = sum;
			}
		}*/
}

bool Conv_Layer::backward(){

	mathematics::set_tensor(activation_grads, 0.0);

	//jedes Element des Gradienten/des Outputs
	#pragma omp parallel for
	for(int grad_z=0; grad_z < activation_grads->getZ(); grad_z++){
		for(int grad_y=0; grad_y < activation_grads->getY(); grad_y++){
			for(int grad_x=0; grad_x < activation_grads->getX(); grad_x++){

				int start_x_rec=0;
				int stop_x_rec=x_receptive-1;
				if(grad_x < x_receptive-1) stop_x_rec = grad_x;
				else if(grad_x > activation->getX()-x_receptive) start_x_rec = x_receptive + grad_x - activation->getX();

				int start_y_rec=0;
				int stop_y_rec=y_receptive-1;
				if(grad_y < y_receptive-1) stop_y_rec = grad_y;
				else if(grad_y > activation->getY()-y_receptive) start_y_rec = y_receptive + grad_y - activation->getY();

				//über verschiedene Features
				for(int z_pos=grad_z; z_pos < weight->getZ(); z_pos+=activation_grads->getZ()){
					//Ueber Gewichte iterieren und in float-Array speichern
					for(int y_rec = start_y_rec; y_rec <= stop_y_rec ; y_rec++){
						for(int x_rec = start_x_rec; x_rec <= stop_x_rec ; x_rec++){
							activation_grads->getArray(grad_z, grad_y)[grad_x]  += output_grads->getArray(z_pos/activation->getZ(), grad_y-y_rec)[grad_x-x_rec] * weight->getArray(z_pos, y_rec)[x_rec];
						}
					}
					activation_grads->getArray(grad_z, grad_y)[grad_x] *= mathematics::sigmoid_backward_derivated_once(activation->getArray(grad_z, grad_y)[grad_x]);
				}
			}
		}
	}

	//weight_grads & bias_grads
	#pragma omp parallel for
	for(int z_pos=0; z_pos < output->getZ(); z_pos++){

		for(int y_pos=0; y_pos < output->getY(); y_pos++){
			for(int x_pos=0; x_pos < output->getX(); x_pos++){

				//Bias hinzufuegen
				bias_grads->getArray()[z_pos] += output_grads->getArray(z_pos, y_pos)[x_pos];

				//Weight hinzufuegen
				for(int weight_z=z_pos; weight_z < weight->getZ(); weight_z+=activation->getZ()){
					for(int weight_x=0; weight_x < weight->getX(); weight_x++){
						for(int weight_y=0; weight_y < weight->getY(); weight_y++){
							weight_grads->getArray(weight_z,weight_y)[weight_x] += output_grads->getArray(z_pos, y_pos)[x_pos] * activation->getArray(weight_z%activation->getZ(), y_pos+weight_y)[x_pos+weight_x];
						}
					}
				}
			}
		}
	}

	return true;
}

bool Conv_Layer::fix(int batch_size, float training_rate){
	#pragma omp parallel for
	for(int feature_map=0; feature_map<no_feature_maps; feature_map++){
		bias->getArray()[feature_map] -= training_rate/batch_size * bias_grads->getArray()[feature_map];
		for(int x_rec=0; x_rec < x_receptive; x_rec++){
			for(int y_rec=0; y_rec < y_receptive; y_rec ++){
				weight->getArray(x_rec, y_rec)[feature_map] -= training_rate/(batch_size) * weight_grads->getArray()[feature_map];
			}
		}
	}

	mathematics::set_tensor(weight_grads, 0.0);
	mathematics::set_tensor(bias_grads, 0.0);
	return true;
}
