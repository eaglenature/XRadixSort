/*
 * xradixsort.cu
 *
 *  Created on: Mar 23, 2013
 *      Author: eaglenature@gmail.com
 */

#include "xtester.h"
#include "xdevice.h"
#include "xtimer.h"
#include "xradixsort.h"


// Map number of elements per load to type
template <int ELEMENTS_PER_LOAD> struct LoadTraits;
template <> struct LoadTraits<1> { typedef uint  Type; };
template <> struct LoadTraits<2> { typedef uint2 Type; };
template <> struct LoadTraits<4> { typedef uint4 Type; };


template <typename Key>
struct RadixSortStorage {

    Key*   d_inputKeys;
    Key*   d_outputKeys;
    uint*  d_spine;
    uint   ARRAY_SIZE;

    inline void InitDeviceStorage(const Key* const inKeys) {
        checkCudaErrors(cudaMalloc((void**) &d_inputKeys,  sizeof(Key) * ARRAY_SIZE));
        checkCudaErrors(cudaMalloc((void**) &d_outputKeys, sizeof(Key) * ARRAY_SIZE));
        checkCudaErrors(cudaMemcpy(d_inputKeys, inKeys,    sizeof(Key) * ARRAY_SIZE, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(d_outputKeys, 0,        sizeof(Key) * ARRAY_SIZE));
    }

    inline void SyncDeviceStorage(Key* outKeys) {
        if (outKeys)      checkCudaErrors(cudaMemcpy(outKeys, d_outputKeys, sizeof(Key) * ARRAY_SIZE, cudaMemcpyDeviceToHost));
        if (d_inputKeys)  checkCudaErrors(cudaFree(d_inputKeys));
        if (d_outputKeys) checkCudaErrors(cudaFree(d_outputKeys));
        if (d_spine)      checkCudaErrors(cudaFree(d_spine));
    }

    inline void Swap() {
        Key* p = d_inputKeys;
        d_inputKeys = d_outputKeys;
        d_outputKeys = p;
    }

    explicit RadixSortStorage(uint size)
    : d_inputKeys(0)
    , d_outputKeys(0)
    , d_spine(0)
    , ARRAY_SIZE(size) {
    }

    ~RadixSortStorage() {
    }
};


template <typename Key>
class RadixSortEnactor {
private:

    uint  _num_elements;
    uint  _num_spine_elements;
    uint  _blocks;
    uint  _threads;
    uint  _tiles;
    uint  _ptiles;

    template<int PASS, int RADIX_DIGITS, int BITS>
    inline cudaError_t DistributionSortPass(RadixSortStorage<Key>& storage);

public:

    explicit RadixSortEnactor(uint num_elements);
    inline cudaError_t Enact(RadixSortStorage<Key>& storage);
};


template <typename Key>
RadixSortEnactor<Key>::RadixSortEnactor(uint num_elements)
: _num_elements(num_elements) {

    const int N = _num_elements;
    const int C = CTAs;
    const int T = NUM_THREADS;
    const int B = (N/(T*C));

    _blocks  = C;
    _threads = T;
    _tiles   = B;
    _ptiles  = B/4;

    const int RADIX_DIGITS = 4;
    _num_spine_elements = RADIX_DIGITS * CTAs + RADIX_DIGITS;

    printf("GridDim:   %d\n", _blocks);
    printf("BlockDim:  %d\n", _threads);
}


template <typename Key>
cudaError_t
RadixSortEnactor<Key>::Enact(RadixSortStorage<Key>& storage) {

    checkCudaErrors(cudaMalloc((void**) &storage.d_spine, sizeof(uint) * _num_spine_elements));

    DistributionSortPass< 0, 4, 2>(storage);
    DistributionSortPass< 1, 4, 2>(storage);
    DistributionSortPass< 2, 4, 2>(storage);
    DistributionSortPass< 3, 4, 2>(storage);
    DistributionSortPass< 4, 4, 2>(storage);
    DistributionSortPass< 5, 4, 2>(storage);
    DistributionSortPass< 6, 4, 2>(storage);
    DistributionSortPass< 7, 4, 2>(storage);
    DistributionSortPass< 8, 4, 2>(storage);
    DistributionSortPass< 9, 4, 2>(storage);
    DistributionSortPass<10, 4, 2>(storage);
    DistributionSortPass<11, 4, 2>(storage);
    DistributionSortPass<12, 4, 2>(storage);
    DistributionSortPass<13, 4, 2>(storage);
    DistributionSortPass<14, 4, 2>(storage);
    DistributionSortPass<15, 4, 2>(storage);

    storage.Swap();

    return cudaSuccess;
}

template <typename Key>
template <int PASS, int RADIX_DIGITS, int BITS>
cudaError_t
RadixSortEnactor<Key>::DistributionSortPass(RadixSortStorage<Key>& storage) {

    typedef typename LoadTraits<4>::Type KeyLoadType;

    UpsweepReduceKernel<KeyLoadType, PASS, RADIX_DIGITS, BITS><<<_blocks, _threads>>>(
            storage.d_spine,
            (KeyLoadType*)storage.d_inputKeys,
            _ptiles,
            _num_elements);
    synchronizeIfEnabled("UpsweepReduceKernel");


    SpineKernel<RADIX_DIGITS><<<RADIX_DIGITS/4, _threads>>>(
            storage.d_spine,
            _num_spine_elements);
    synchronizeIfEnabled("SpineKernel");


    DownsweepScanKernel<Key, PASS, RADIX_DIGITS, BITS><<<_blocks, _threads>>>(
            storage.d_outputKeys,
            storage.d_inputKeys,
            storage.d_spine,
            _tiles,
            _num_elements);
    synchronizeIfEnabled("DownsweepScanKernel");

    storage.Swap();
    return cudaSuccess;
}


int main(int argc, char** argv) {

    typedef uint KeyType;

    CudaDevice device;
    device.Init();

    RadixSortTester<KeyType> tester(argc, argv, device);
    tester.InitSample();

    RadixSortStorage<KeyType> deviceStorage(tester.array_size);
    deviceStorage.InitDeviceStorage(tester.sample.data());

    RadixSortEnactor<KeyType> radixSorter(tester.array_size);

    CudaDeviceTimer deviceTimer;

    deviceTimer.Start();
    radixSorter.Enact(deviceStorage);
    deviceTimer.Stop();

    tester.ShowStats(deviceTimer.ElapsedTime());
    deviceStorage.SyncDeviceStorage(tester.result.data());
    tester.CompareResults();
}
