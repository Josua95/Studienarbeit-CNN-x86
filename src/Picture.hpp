/*
 * Picture.h
 *
 *  Created on: 03.12.2017
 *      Author: Florian Schmidt, Josua Benz
 */

#ifndef PICTURE_HPP_
#define PICTURE_HPP_

#include <string>

#define INPUT_SIZE 		784
#define OUTPUT_SIZE 	10

class Picture {
private:
	float input_data[INPUT_SIZE] = {0.0};
	float output_data[OUTPUT_SIZE] = {0.0};
public:
	Picture(std::string *line);
	Picture();
	virtual ~Picture();
	float *get_input(void);
	float *get_output(void);
};

#endif /* PICTURE_HPP_ */
