#include "mlutility.h"

#include <cstdlib>
#include <cmath>

double ml::randu(double min, double max)
{
    double z = static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * (max - min);
    return z + min;
}

double ml::randn(double mean, double stddev)
{
    double z = sqrt(-2.0 * log(ml::randu(0.0, 1.0))) * sin(2.0 * M_PI * randu(0.0, 1.0));
    return mean + stddev * z;
}
