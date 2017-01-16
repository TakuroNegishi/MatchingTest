#ifndef MOVING_AVERAGE_FILTER_H
#define MOVING_AVERAGE_FILTER_H

#include "Filter.h"

class MovingAverageFilter : public Filter
{
private:
	const static int WIN_SIZE = 7; // •½‹Ï‘‹ƒTƒCƒY
	float data[WIN_SIZE]; // 
	float total;
	int pointer;
	bool isFilledDataArray;
public:
	MovingAverageFilter();
	~MovingAverageFilter();
	float update(const float measurement);
	void clear();
};

#endif // MOVING_AVERAGE_FILTER_H