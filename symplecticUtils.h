
#ifndef SYMPLECTIC_UTILS_H
#define SYMPLECTIC_UTILS_H

/****
 * In C++11, in-class member initializers are allowed,
 * but basically act the same as initializing in a member initialization list. 
 * Therefore, the size of the array must be explicitly stated.
 ****/

namespace SymplecticUtils {
  
  // from J. Chem. Phys. 104, 7099 (1996)
  struct {
    const double a[6] = { 0.2167979108466032, -0.0283101143283301, 0.3901418904713324,
			 -0.2414087476423302,  0.5908564573813148, 0.0719226032714098 };
    
    const double b[6] = { 0.0719226032714098,  0.5908564573813148, -0.2414087476423302, 
			  0.3901418904713324, -0.0283101143283301,  0.2167979108466032 };
    
    const int size = 6;
    
  } coeffients_m6_n4;
  
  // from Computer Physics Communications 98, 1 (1996)
  struct {
    
    const double a[6] = {  0.216797938, -0.028310106, 0.390141840, 
			  -0.241408784,  0.590856492, 0.071922619 };
    
    const double b[6] = { 0.071922619,  0.590856492, -0.241408784, 
			  0.390141840, -0.028310106,  0.216797938 };
    
    const int size = 6;
    
  } coeffients_m6_n4_1;
}

#endif /* SYMPLECTIC_UTILS_H */
