/*
 * PictureContainer.h
 *
 *  Created on: 03.12.2017
 *      Author: Florian Schmidt, Josua Benz
 */

#ifndef PICTURECONTAINER_HPP_
#define PICTURECONTAINER_HPP_

#include "Picture.hpp"

#define PICS_PER_FILE 1000

class PictureContainer {
private:
	Picture images[PICS_PER_FILE] = { Picture() };
	int next_index;
	int file_index;
	int num_of_files;
	std::string foldername;
	void load_pictures();
public:
	PictureContainer(std::string foldername, int num_of_files);
	virtual ~PictureContainer();
	Picture *get_nextpicture(void);
};

#endif /* PICTURECONTAINER_HPP_ */
