#ifndef VECTOR_COMMON_HPP
#define VECTOR_COMMON_HPP

#ifndef __NVCC__
#   ifdef USE_AVX
#       include <immintrin.h>
//#       define USE_AVX_VECTORS
#   endif
#endif

#endif //VECTOR_COMMON_HPP