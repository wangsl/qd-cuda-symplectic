
#ifndef _GENUTILS_H
#define _GENUTILS_H

#include <sys/time.h>
#include "cudaUtils.h"

inline double get_time_now_in_secs()
{
  struct timeval t;
  insist(gettimeofday(&t, NULL) == 0);
  return t.tv_sec + t.tv_usec*1.0e-6;
}

#endif
