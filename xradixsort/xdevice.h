/*
 * xdevice.h
 *
 *  Created on: Mar 23, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef XDEVICE_H_
#define XDEVICE_H_

#include "xerror.h"

class CudaDevice {
public:

    float peakBandwidth;
    int   multiProcessorCount;

    CudaDevice ()
    : currentDeviceID(0)
    , peakBandwidth(0.0f)
    , multiProcessorCount(0) {
    }

    ~CudaDevice() {
        checkCudaErrors(cudaDeviceReset());
    }

    void Init() {
        cudaError_t error;
        cudaDeviceProp deviceProp;
        error = cudaGetDevice(&currentDeviceID);
        if (error != cudaSuccess) {
            fprintf(stderr, "cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
            exit(1);
        }
        error = cudaGetDeviceProperties(&deviceProp, currentDeviceID);
        if (deviceProp.computeMode == cudaComputeModeProhibited) {
            fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, "
                    "no threads can use ::cudaSetDevice().\n");
            exit(1);
        }
        if (error != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
            exit(1);
        } else {
            printf("\nCudaDevice %d:  %s\nCompute cap:  %d.%d\n",
                currentDeviceID,
                deviceProp.name,
                deviceProp.major,
                deviceProp.minor);
        }
        multiProcessorCount = deviceProp.multiProcessorCount;
        peakBandwidth = 2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8.0)/1.0e6;
    }

private:
    int currentDeviceID;
    CudaDevice(const CudaDevice& device);
    CudaDevice& operator=(const CudaDevice& device);
};


#endif /* XDEVICE_H_ */
