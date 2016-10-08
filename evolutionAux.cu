
#ifndef EVOLUTION_AUX_H
#define EVOLUTION_AUX_H

#ifdef printf
#undef printf
#endif

static __global__ void _print_constant_memory()
{
  printf(" %f %f %f %d\n", r1_dev.left, r1_dev.dr, r1_dev.mass, r1_dev.n);
  printf(" %f %f %f %d\n", r2_dev.left, r2_dev.dr, r2_dev.mass, r2_dev.n);
  for(int i = 0; i < 500; i+=10) 
    printf("%d %18.15f %18.15f\n", i+1, r1_dev.dump[i], r2_dev.dump[i]);
}

#endif /* EVOLUTION_AUX_H */
