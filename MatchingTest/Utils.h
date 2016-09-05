#ifndef UTILS_H
#define UTILS_H

#include <math.h>

namespace Utils {
	float getDistance(const int pt1x, const int pt1y, const int pt2x, const int pt2y)
	{
		float dx = pt2x - pt1x;
		float dy = pt2y - pt1y;
		return sqrt(dx * dx + dy * dy);
	}
}

#endif // UTILS_H
