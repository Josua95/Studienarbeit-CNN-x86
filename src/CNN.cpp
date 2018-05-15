/*
 * CNN.cpp
 *
 *  Created on: 09.03.2018
 *      Author: josua
 */
#include <vector>
#include <iostream>
#include <math.h>
#include "ConvLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "PictureContainer.hpp"
#include "Mathematics.hpp"

#define TRAINING_RATE 0.01
#define BATCH_SIZE 10
#define EPOCH 100
#define ITERATIONS 100000

union Layer{
	Conv_Layer *conv_layer;
	MaxPooling_Layer *max_pooling_layer;
	FullyConnected_Layer *fully_connected_layer;
};

int main(int argc, char **argv) {
	//seed for random number
	srand (static_cast <unsigned> (time(0)));

	std::vector<Layer*> *layers = new std::vector<Layer*>;
	Layer layer1;
	layer1.conv_layer = new Conv_Layer(5,5,1,32);
	Layer layer2;
	layer2.max_pooling_layer = new MaxPooling_Layer(2,2);
	Layer layer3;
	layer3.conv_layer = new Conv_Layer(5,5,1,64);
	Layer layer4;
	layer4.max_pooling_layer = new MaxPooling_Layer(2,2);
	Layer layer5;
	layer5.fully_connected_layer = new FullyConnected_Layer(1024);
	Layer layer6;
	layer6.fully_connected_layer = new FullyConnected_Layer(10);

	layers->push_back(&layer1);
	layers->push_back(&layer2);
	layers->push_back(&layer3);
	layers->push_back(&layer4);
	layers->push_back(&layer5);
	layers->push_back(&layer6);

	Tensor *new_picture = new Tensor(28,28,1);

	//Netzwerk erstellen
	for(unsigned int layer_index=0;layer_index<layers->size(); layer_index++){
		switch(layer_index){
		case 0:
			layers->at(layer_index)->conv_layer->generate(new_picture, new_picture);
			break;
		case 1:
			layers->at(layer_index)->max_pooling_layer->generate(layers->at(layer_index-1)->conv_layer->output, layers->at(layer_index-1)->conv_layer->output_grads);
			break;
		case 2:
			layers->at(layer_index)->conv_layer->generate(layers->at(layer_index-1)->max_pooling_layer->output, layers->at(layer_index-1)->max_pooling_layer->output_grads);
			break;
		case 3:
			layers->at(layer_index)->max_pooling_layer->generate(layers->at(layer_index-1)->conv_layer->output, layers->at(layer_index-1)->conv_layer->output_grads);
			break;
		case 4:
			layers->at(layer_index)->fully_connected_layer->generate(layers->at(layer_index-1)->max_pooling_layer->output, layers->at(layer_index-1)->max_pooling_layer->output_grads);
			break;
		case 5:
			layers->at(layer_index)->fully_connected_layer->generate(layers->at(layer_index-1)->fully_connected_layer->output, layers->at(layer_index-1)->fully_connected_layer->output_grads);
			break;
		default: break;
		}

	}

	PictureContainer *train_picture_container = new PictureContainer("./train", 1);

	//FORWARD
	for(int n=0; n<ITERATIONS; n++){
		int hit=0;
		int hit1[10] = {0,0,0,0,0,0,0,0,0,0};
		int hit2[10] = {0,0,0,0,0,0,0,0,0,0};
		float cost=0.0;
		//1 Epoche

		for(int i=0; i<EPOCH*BATCH_SIZE; i++){

			Picture *picture = train_picture_container->get_nextpicture();
			for(int y=0; y<new_picture->getY(); y++){
				for(int x=0; x<new_picture->getX(); x++){
					new_picture->getArray(0,y)[x] = picture->get_input()[y*new_picture->getY()+x];
				}
			}


			/*std::cout << std::endl << "Output:" << std::endl;
			for(int i=0; i<10; i++){
				std::cout << picture->get_output()[i] << " ";
			}
			std::cout << std::endl;*/

			for(unsigned int layer_index=0;layer_index < layers->size(); layer_index++){
				switch(layer_index){
				case 0:
					layers->at(layer_index)->conv_layer->forward();
					//mathematics::printTensor(layers->at(layer_index)->conv_layer->output);
					//mathematics::printTensor(layers->at(layer_index)->conv_layer->weight);
					break;
				case 1:
					layers->at(layer_index)->max_pooling_layer->forward();
					//mathematics::printTensor(layers->at(layer_index)->max_pooling_layer->output);
					break;
				case 2:
					layers->at(layer_index)->conv_layer->forward();
					//mathematics::printTensor(layers->at(layer_index)->conv_layer->output);
					//mathematics::printTensor(layers->at(layer_index)->conv_layer->weight);
					break;
				case 3:
					layers->at(layer_index)->max_pooling_layer->forward();
					//mathematics::printTensor(layers->at(layer_index)->max_pooling_layer->output);
					break;
				case 4:
					layers->at(layer_index)->fully_connected_layer->forward();
					//mathematics::printTensor(layers->at(layer_index)->fully_connected_layer->weight);
					//mathematics::printTensor(layers->at(layer_index)->fully_connected_layer->output);
					break;
				case 5:
					layers->at(layer_index)->fully_connected_layer->forward();
					//mathematics::printTensor(layers->at(layer_index)->fully_connected_layer->weight);
					//mathematics::printTensor(layers->at(layer_index)->fully_connected_layer->output);
					break;
				default: break;
				}
			}
			float soft_output[10];
			mathematics::softmax(layers->back()->fully_connected_layer->output->getArray(), soft_output, 10);
			/*for(int i=0; i<10; i++){
				layers->at(4)->fully_connected_layer->output->getArray()[i] = soft_output[i];
			}*/
			//BACKWARD
			for(int i=0; i < 10; i++){
				if(picture->get_output()[i] == 1.0f){
					layers->back()->fully_connected_layer->output_grads->getArray()[i] = layers->back()->fully_connected_layer->output->getArray()[i] -1.0f;
					cost += pow( 1.0 - layers->back()->fully_connected_layer->output->getArray()[i], 2);
					layers->back()->fully_connected_layer->output_grads->getArray()[i] *= mathematics::sigmoid_backward_derivated_once(layers->back()->fully_connected_layer->output->getArray()[i]);
					//std::cout << 1.0f - layers->at(4)->fully_connected_layer->output->getArray()[i] << " ";
				}
				else{
					layers->back()->fully_connected_layer->output_grads->getArray()[i] = layers->back()->fully_connected_layer->output->getArray()[i];
					cost += pow( layers->back()->fully_connected_layer->output->getArray()[i] , 2);
					layers->back()->fully_connected_layer->output_grads->getArray()[i] *= mathematics::sigmoid_backward_derivated_once(layers->back()->fully_connected_layer->output->getArray()[i]);
					//std::cout << -layers->at(4)->fully_connected_layer->output->getArray()[i] << " ";
				}
			}
			//std::cout << std::endl;



			for(int layer_index=layers->size()-1;layer_index >= 0; layer_index--){
				switch(layer_index){
				case 0:
					layers->at(layer_index)->conv_layer->backward();
					//mathematics::printTensor(layers->at(layer_index)->conv_layer->node);
					//mathematics::printTensor(layers->at(layer_index)->conv_layer->weight);
					break;
				case 1:
					layers->at(layer_index)->max_pooling_layer->backward();
					//mathematics::printTensor(layers->at(layer_index)->max_pooling_layer->node);
					break;
				case 2:
					layers->at(layer_index)->conv_layer->backward();
					break;
				case 3:
					layers->at(layer_index)->max_pooling_layer->backward();
					break;
				case 4:
					layers->at(layer_index)->fully_connected_layer->backward();
					break;
				case 5:
					layers->at(layer_index)->fully_connected_layer->backward();
					break;
				default: break;
				}
			}

			float *output=layers->back()->fully_connected_layer->output->getArray();
			//std::cout << "Forward " << i << " " << output[0] << " " << output[1] << " " << output[2] << " " << output[3] << " " << output[4] << " " << output[5] << " " << output[6] << " " << output[7] << " " << output[8] << " " << output[9] << std::endl;

			//ermitteln, ob Output passt
			int max_index=0;
			float max_value=0;
			for(int i=0; i < layers->back()->fully_connected_layer->output->getX();i++){
				if(max_value < output[i]){
					max_value = output[i];
					max_index = i;
				}
			}
			if(picture->get_output()[max_index] == 1){
				hit++;
				hit1[max_index]++;
			}
			for(int i=0; i< 10; i++){
				if(picture->get_output()[i] == 1){
					hit2[i]++;
				}
			}


			//FIX
			if(i%BATCH_SIZE+1 == BATCH_SIZE){
				for(unsigned int layer_index=0;layer_index < layers->size(); layer_index++){
					switch(layer_index){
					case 0:
						//mathematics::printTensor(layers->at(layer_index)->conv_layer->weight_grads);
						layers->at(layer_index)->conv_layer->fix(BATCH_SIZE, TRAINING_RATE);
						break;
					case 1:
						//MaxPoolingLayer
						break;
					case 2:
						//mathematics::printTensor(layers->at(layer_index)->conv_layer->weight_grads);
						layers->at(layer_index)->conv_layer->fix(BATCH_SIZE, TRAINING_RATE);
						break;
					case 3:
						//MaxPoolingLayer
						break;
					case 4:
						//mathematics::printTensor(layers->at(layer_index)->fully_connected_layer->weight_grads);
						layers->at(layer_index)->fully_connected_layer->fix(BATCH_SIZE, TRAINING_RATE);
						break;
					case 5:
						//mathematics::printTensor(layers->at(layer_index)->fully_connected_layer->weight_grads);
						layers->at(layer_index)->fully_connected_layer->fix(BATCH_SIZE, TRAINING_RATE);
						break;
					default: break;
					}
				}
			}
		}

		std::cout << "Finished Epoche " << n << std::endl;
		std::cout << "Rate: " << (double)hit/((double)BATCH_SIZE*EPOCH) << std::endl;
		std::cout << "Cost: " << cost << std::endl;
		for(int i=0; i<10; i++){
			std::cout << hit1[i] << " ";
		}
		std::cout << std::endl;
		for(int i=0; i<10; i++){
			std::cout << hit2[i] << " ";
		}
		std::cout << std::endl;
	}
}


