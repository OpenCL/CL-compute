__kernel void saxpy(__global float *x,
                    __global float *y,
                    const float a)
{
    const uint i = get_global_id(0);
    y[i] = a * x[i] + y[i];
};