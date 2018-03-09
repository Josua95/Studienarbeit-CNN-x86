/*
 * CNN.cpp
 *
 *  Created on: 09.03.2018
 *      Author: josua
 */
#include <vector>
#include <iostream>
#include "InputLayer.hpp"
#include "ConvLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "PictureContainer.hpp"

union Layer{
	Input_Layer *input_layer;
	Conv_Layer *conv_layer;
	MaxPooling_Layer *max_pooling_layer;
	FullyConnected_Layer *fully_connected_layer;
};

int main(int argc, char **argv) {
	std::vector<Layer*> *layers = new std::vector<Layer*>;
	Layer layer1;
	Input_Layer *inputlayer = new Input_Layer(28,28);
	layer1.input_layer = inputlayer;
	Layer layer2;
	layer2.conv_layer = new Conv_Layer(5,5,1,60);
	Layer layer3;
	layer3.max_pooling_layer = new MaxPooling_Layer(2,2);
	Layer layer4;
	layer4.conv_layer = new Conv_Layer(5,5,1,80);
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
			layers->at(layer_index)->conv_layer->generate(layers->at(layer_index-1)->input_layer->getNode());
			break;
		case 2:
			layers->at(layer_index)->max_pooling_layer->generate(layers->at(layer_index-1)->conv_layer);
			break;
		case 3:
			layers->at(layer_index)->conv_layer->generate(layers->at(layer_index-1)->max_pooling_layer->getNodeTensor());
			break;
		case 4:
			layers->at(layer_index)->max_pooling_layer->generate(layers->at(layer_index-1)->conv_layer);
			break;
		case 5:
			layers->at(layer_index)->fully_connected_layer->generate(layers->at(layer_index-1)->max_pooling_layer->getNodeTensor());
			break;
		default: break;
		}

	}

	//Forward
	PictureContainer *train_picture_container = new PictureContainer("./train", 1);

	for(int i=0; i<1000; i++){
		Picture *picture = train_picture_container->get_nextpicture();

		for(unsigned int layer_index=0;layer_index<layers->size(); layer_index++){
			switch(layer_index){
			case 0:
				layers->at(layer_index)->input_layer->forward(picture->get_input());
				break;
			case 1:
				layers->at(layer_index)->conv_layer->forward(layers->at(layer_index-1)->input_layer->getNode());
				break;
			case 2:
				layers->at(layer_index)->max_pooling_layer->forward(layers->at(layer_index-1)->conv_layer->getNodeTensor());
				break;
			case 3:
				layers->at(layer_index)->conv_layer->forward(layers->at(layer_index-1)->max_pooling_layer->getNodeTensor());
				break;
			case 4:
				layers->at(layer_index)->max_pooling_layer->forward(layers->at(layer_index-1)->conv_layer->getNodeTensor());
				break;
			case 5:
				layers->at(layer_index)->fully_connected_layer->forward(layers->at(layer_index-1)->max_pooling_layer->getNodeTensor());
				break;
			default: break;
			}
		}
		std::cout << "Forward" << i << std::endl;

	}


}


