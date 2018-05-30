/*
 * PictureContainer.cpp
 *
 *  Created on: 03.12.2017
 *      Author: Florian Schmidt, Josua Benz
 */

#include "PictureContainer.hpp"

#include <fstream>


/**
 * Constructor für den PictureContainer
 * foldername: Pfad zum Ordner mit den Bildern
 * num_fo_files: Anzahl der .csv-Dateien mit Bildern im Ordner
 */
PictureContainer::PictureContainer(std::string foldername, int num_of_files)
{
	this->next_index = -1;
	this->file_index = 0;
	this->foldername = foldername;
	this->num_of_files = num_of_files;
	load_pictures();
}

PictureContainer::~PictureContainer() {

}

/**
 * Laden der Bilder aus .csv-Dateien innerhalb des Ordners
 */
void PictureContainer::load_pictures() {
	std::string csv_file = this->foldername + "/" + std::to_string(this->file_index) + ".csv";
	std::ifstream infile(csv_file);
	for(int i=0; i<PICS_PER_FILE; i++)
	{
		std::string line;
		std::getline(infile,line);
		this->images[i] = Picture(&line);
	}
}

/**
 * nächstes Bild wird geladen
 * falls Ende einer Datei erreicht, wird erstes Bild aus nächstes Datei genommen
 */
Picture * PictureContainer::get_nextpicture(void)
{
	next_index++;
	if(next_index >= PICS_PER_FILE)
	{
		next_index=0;
		file_index++;
		if(file_index >= num_of_files)
		{
			file_index = 0;
		}
		load_pictures();
	}
	return this->images + next_index;
}
