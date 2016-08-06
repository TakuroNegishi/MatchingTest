#ifndef FILTER_H
#define FILTER_H

#include <iostream>

class Filter
{
public:
	Filter(){};
	virtual ~Filter(){
		std::cout << "Filter destractor" << std::endl;
	};
	/* @brief フィルタ適用後の値を返す
	@param measurement 観測値
	@return 観測値をフィルタに通した後の値 */
	virtual float update(const float measurement) = 0;
	virtual void clear() = 0;
};

#endif // FILTER_H