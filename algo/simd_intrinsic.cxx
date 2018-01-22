/**
 * SIMD Intrinsics 
 */

#include <smmintrin.h>
#include <vector>
#include <iostream>

using namespace std;

void fcond(float *x, size_t n) {
    int i;
    for (i = 0; i < n; ++i) {
        if (x[i] > 0.5)
            x[i] += 1.;
        else 
            x[i] -= 1.;
    }
}

void fcond_simd(float *a, size_t n) {
    int i;
    __m128 vt, vr, vtp1, vtm1, vmask, ones, thresholds;
    ones = _mm_set1_ps(1.);
    thresholds = _mm_set1_ps(0.5);
    for (i = 0; i < n; i += 4) {
        vt = _mm_load_ps(a + i);
        vmask = _mm_cmpgt_ps(vt, thresholds);
        vtp1 = _mm_add_ps(vt, ones);
        vtm1 = _mm_sub_ps(vt, ones);
        vr = _mm_blendv_ps(vtm1, vtp1, vmask);
        _mm_store_ps(a + i, vr);
    }
}

int main() {
    auto vec = vector<float> {1,1,2,3,4,5,6,7};
    fcond(&vec[0], vec.size());
    fcond_simd(&vec[0], vec.size());
}
