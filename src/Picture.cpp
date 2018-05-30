/*
 * Picture.cpp
 *
 *  Created on: 03.12.2017
 *      Author: Florian
 */

#include "Picture.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>



Picture::Picture()
{

}

/**
 * Constructor
 * line: String mit den Werten eines Bildes in .csv-Format
 */
Picture::Picture(std::string *line)
{
	std::stringstream          lineStream(*line);
	std::string                cell;

	for(int i=0; i<INPUT_SIZE; i++)
	{
		std::getline(lineStream,cell, ',');
		this->input_data[i] = std::stof(cell);
	}
	for(int i=0; i<OUTPUT_SIZE; i++)
	{
		std::getline(lineStream,cell, ',');
		this->output_data[i] = (float)std::stod(cell);
	}
}

Picture::~Picture() { }

/**
 * return: Zeiger auf die Elemente des Bildes
 */
float *Picture::get_input(void)
{
	return this->input_data;
}

/**
 * return: Zeiger auf die richtige Klassifizierung der Bilder
 */
float *Picture::get_output(void)
{
	return this->output_data;
}
