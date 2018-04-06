/*
 * CNN.cpp
 *
 *  Created on: 09.03.2018
 *      Author: josua
 */
#include <vector>
#include <iostream>
#include <math.h>
#include "InputLayer.hpp"
#include "ConvLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "PictureContainer.hpp"
#include "Mathematics.hpp"

union Layer{
	Input_Layer *input_layer;
	Conv_Layer *conv_layer;
	MaxPooling_Layer *max_pooling_layer;
	FullyConnected_Layer *fully_connected_layer;
};

int main(int argc, char **argv) {
	//seed for random number
	srand (static_cast <unsigned> (time(0)));

	std::vector<Layer*> *layers = new std::vector<Layer*>;
	Layer layer1;
	Input_Layer *inputlayer = new Input_Layer(28,28);
	layer1.input_layer = inputlayer;
	Layer layer2;
	layer2.conv_layer = new Conv_Layer(5,5,1,6);
	Layer layer3;
	layer3.max_pooling_layer = new MaxPooling_Layer(2,2);
	Layer layer4;
	layer4.conv_layer = new Conv_Layer(5,5,1,16);
	Layer layer5;
	layer5.max_pooling_layer = new MaxPooling_Layer(2,2);
	Layer layer6;
	layer6.fully_connected_layer = new FullyConnected_Layer(10);

	layers->push_back(&layer1);
	layers->push_back(&layer2);
	layers->push_back(&layer3);
	layers->push_back(&layer4);
	layers->push_back(&layer5);
	layers->push_back(&layer6);

	//Netzwerk erstellen
	for(unsigned int layer_index=0;layer_index<layers->size(); layer_index++){
		switch(layer_index){
		case 0:
			break;
		case 1:
			//TODO kein pre_grads bei input_layer
			layers->at(layer_index)->conv_layer->generate(layers->at(layer_index-1)->input_layer->output, layers->at(layer_index-1)->input_layer->output);
			break;
		case 2:
			layers->at(layer_index)->max_pooling_layer->generate(layers->at(layer_index-1)->conv_layer->output, layers->at(layer_index-1)->conv_layer->grads);
			break;
		case 3:
			layers->at(layer_index)->conv_layer->generate(layers->at(layer_index-1)->max_pooling_layer->output, layers->at(layer_index-1)->max_pooling_layer->grads);
			break;
		case 4:
			layers->at(layer_index)->max_pooling_layer->generate(layers->at(layer_index-1)->conv_layer->output, layers->at(layer_index-1)->conv_layer->grads);
			break;
		case 5:
			layers->at(layer_index)->fully_connected_layer->generate(layers->at(layer_index-1)->max_pooling_layer->output, layers->at(layer_index-1)->max_pooling_layer->grads);
			break;
		default: break;
		}

	}

	PictureContainer *train_picture_container = new PictureContainer("./train", 1);

	//FORWARD
	for(int i=0; i<999; i++){
		Picture *picture = train_picture_container->get_nextpicture();

		/*std::cout << std::endl << "Input:" << std::endl;
		for(int i=0; i<28*28; i++){
			std::cout << picture->get_input()[i] << " ";
		}
		std::cout << std::endl << "Output:" << std::endl;
		for(int i=0; i<10; i++){
			std::cout << picture->get_output()[i] << " ";
		}
		std::cout << std::endl;*/

		for(unsigned int layer_index=0;layer_index<layers->size(); layer_index++){
			switch(layer_index){
			case 0:
				layers->at(layer_index)->input_layer->forward(picture->get_input());
				//mathematics::printTensor(layers->at(layer_index)->input_layer->node));
				break;
			case 1:
				layers->at(layer_index)->conv_layer->forward();
				//mathematics::printTensor(layers->at(layer_index)->conv_layer->output);
				//mathematics::printTensor(layers->at(layer_index)->conv_layer->weight);
				break;
			case 2:
				layers->at(layer_index)->max_pooling_layer->forward();
				//mathematics::printTensor(layers->at(layer_index)->max_pooling_layer->node);
				break;
			case 3:
				layers->at(layer_index)->conv_layer->forward();
				break;
			case 4:
				layers->at(layer_index)->max_pooling_layer->forward();
				break;
			case 5:
				layers->at(layer_index)->fully_connected_layer->forward();
				break;
			default: break;
			}
		}

		//BACKWARD
		for(int i=0; i < 10; i++){
			if(picture->get_output()[i] == 1.0f){
				layers->at(5)->fully_connected_layer->grads->getArray()[i] = 1.0f - layers->at(5)->fully_connected_layer->output->getArray()[i];
				//std::cout << powf(1.0f - layers->at(5)->fully_connected_layer->output->getArray()[i], 2.0f) << " ";
			}
			else{
				layers->at(5)->fully_connected_layer->grads->getArray()[i] = -layers->at(5)->fully_connected_layer->output->getArray()[i];
				//std::cout << powf(1.0f - layers->at(5)->fully_connected_layer->output->getArray()[i], 2.0f) << " ";
			}
		}
		//std::cout << std::endl;


		for(unsigned int layer_index=layers->size()-1;layer_index > 0; layer_index--){
			switch(layer_index){
			case 0:
				//mathematics::printTensor(layers->at(layer_index)->input_layer->node);
				break;
			case 1:
				layers->at(layer_index)->conv_layer->backward();
				//mathematics::printTensor(layers->at(layer_index)->conv_layer->node);
				//mathematics::printTensor(layers->at(layer_index)->conv_layer->weight);
				break;
			case 2:
				layers->at(layer_index)->max_pooling_layer->backward();
				//mathematics::printTensor(layers->at(layer_index)->max_pooling_layer->node);
				break;
			case 3:
				layers->at(layer_index)->conv_layer->backward();
				break;
			case 4:
				layers->at(layer_index)->max_pooling_layer->backward();
				break;
			case 5:
				layers->at(layer_index)->fully_connected_layer->backward();
				break;
			default: break;
			}
		}
		float *output=layers->at(5)->fully_connected_layer->output->getArray();
		std::cout << "Forward " << i << " " << output[0] << " " << output[1] << " " << output[2] << " " << output[3] << " " << output[4] << " " << output[5] << " " << output[6] << " " << output[7] << " " << output[8] << " " << output[9] << std::endl;

		//FIX
		if(i%10 == 9){
			for(unsigned int layer_index=0;layer_index <= 5; layer_index++){
				switch(layer_index){
				case 0:
					//InputLayer
					break;
				case 1:
					//mathematics::printTensor(layers->at(layer_index)->conv_layer->weight_grads);
					layers->at(layer_index)->conv_layer->fix(10, 0.05);
					break;
				case 2:
					//MaxPoolingLayer
					break;
				case 3:
					//mathematics::printTensor(layers->at(layer_index)->conv_layer->weight_grads);
					layers->at(layer_index)->conv_layer->fix(10, 0.05);
					break;
				case 4:
					//MaxPoolingLayer
					break;
				case 5:
					//mathematics::printTensor(layers->at(layer_index)->fully_connected_layer->weight_grads);
					layers->at(layer_index)->fully_connected_layer->fix(10, 0.05);
					break;
				default: break;
				}
			}
		}
	}

	std::cout << "Finished";

}


