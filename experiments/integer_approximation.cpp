#include "raster/composit.h"
#include <cmath>
#include <cstdio>

int div255_true(int a, int b) {
    return (a * b) / 255;
}

int div255_true_rounded(int a, int b) {
    return (a * b + 127) / 255;
}

int div255_by_float(int a, int b) {
    int prod = (a * b);
    float quotient = (float)prod / 255.0f;
    return (int)quotient;
}

int div255_by_float_rounded(int a, int b) {
    int prod = (a * b);
    float quotient = (float)prod / 255.0f;
    return (int)(round(quotient));
}

int div255_by_double(int a, int b) {
    int prod = (a * b);
    double quotient = (double)prod / 255.0;
    return (int)quotient;
}

int div255_by_double_rounded(int a, int b) {
    int prod = (a * b);
    double quotient = (double)prod / 255.0;
    return (int)(round(quotient));
}

int div255_by_approx(int a, int b) {
    int prod = (a * b);
    int quotient = ((prod >> 8) + prod) >> 8;
    return quotient;
}

int div255_by_approx_corrected(int a, int b) {
#define INT_MULT(a,b,t) ( (t) = (a) * (b) + 0x80, ( ( ( (t)>>8 ) + (t) )>>8 ) )
    int t;
    return INT_MULT(a, b, t);
    // int prod = (a * b) + 0x80;
    // int quotient = ((prod >> 8) + prod) >> 8;
    // return quotient;
#undef INT_MULT
}

int div255_by_approx_extra(int a, int b) {
    int prod = (a * b);
    int quotient = ((((prod >> 8) + prod) >> 8) + prod) >> 8;
    return quotient;
}

int div255_by_approx_extra_corrected(int a, int b) {
    int prod = (a * b) + 0x80;
    int quotient = ((((prod >> 8) + prod) >> 8) + prod) >> 8;
    return quotient;
}

#define SHIFTFORDIV255(a) ((((a) >> 8) + a) >> 8)
int div255_borrowed(int a, int b) {
    int tmp = a * b;
    return SHIFTFORDIV255(tmp + (0x80 << 7)) >> 7;
}

int main() {
    for (int a = 0; a <= 255; a++) {
        for (int b = 0; b <= 255; b++) {
            int compare1 = div255_by_double_rounded(a, b);
            int compare2 = div255_borrowed(a, b);
            if (compare1 != compare2) {
                printf("a: %d, b: %d, compare1: %d, compare2: %d\n", a, b, compare1, compare2);
            }
        }
    }

    return 0;
}
