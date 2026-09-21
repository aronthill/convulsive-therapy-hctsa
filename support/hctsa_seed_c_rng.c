/* Seed the C standard-library generator used by hctsa's bundled TSTOOL.
 * No feature calculation is implemented here.
 * MATLAB rng controls a different stream. Call this immediately before
 * TS_Compute and clear corrsum/corrsum2 to reset their static generators too.
 * Compile with: mex hctsa_seed_c_rng.c
 */
#include "mex.h"
#include <stdlib.h>
#include <math.h>
#include <limits.h>
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1 || nlhs != 0 || !mxIsDouble(prhs[0]) ||
        mxIsComplex(prhs[0]) || mxGetNumberOfElements(prhs[0]) != 1)
        mexErrMsgIdAndTxt("hctsa:seedInput", "Provide one real double scalar seed and no output.");
    double value = mxGetScalar(prhs[0]);
    if (!mxIsFinite(value) || value < 0 || value > UINT_MAX || floor(value) != value)
        mexErrMsgIdAndTxt("hctsa:seedRange", "Seed must be an integer from 0 to UINT_MAX.");
    srand((unsigned int)value);
}
