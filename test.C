
#include <iostream>
using namespace std;

#include "symplecticUtils.h"

void my_test()
{
  const double *a = SymplecticUtils::coeffients_m6_n4.a;
  const double *b = SymplecticUtils::coeffients_m6_n4.b;
  const int &size = SymplecticUtils::coeffients_m6_n4.size;
  
  for(int i = 0; i < size; i++)
    std::cout << " " << i << " " << a[i] << " " << b[i] << std::endl;
}
