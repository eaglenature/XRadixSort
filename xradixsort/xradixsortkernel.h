/*
 * xradixsortkernel.h
 *
 *  Created on: Mar 26, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef XRADIXSORTKERNEL_H_
#define XRADIXSORTKERNEL_H_

#include "xconfig.h"

#define FULL_MASK 0xffffffff
typedef unsigned int uint;


template<int BITS> struct DigitMask;
template<typename T> struct UpdateHistogram;
template<typename T, int BITS, int OFFSET> struct ExtractMaskedValue;

template<> struct DigitMask<1> { static const uint value = 0x00000001; };
template<> struct DigitMask<2> { static const uint value = 0x00000003; };
template<> struct DigitMask<4> { static const uint value = 0x0000000F; };
template<> struct DigitMask<8> { static const uint value = 0x000000FF; };

template<> struct UpdateHistogram<uint> {
    __device__
    inline static void apply(uint* const histogram, uint element) {
        ++histogram[element];
    }
};
template<> struct UpdateHistogram<uint2> {
    __device__
    inline static void apply(uint* const histogram, const uint2& element) {
        ++histogram[element.x];
        ++histogram[element.y];
    }
};
template<> struct UpdateHistogram<uint4> {
    __device__
    inline static void apply(uint* const histogram, const uint4& element) {
        ++histogram[element.x];
        ++histogram[element.y];
        ++histogram[element.z];
        ++histogram[element.w];
    }
};

template<int BITS, int OFFSET> struct ExtractMaskedValue<uint,  BITS, OFFSET> {
    __device__
    inline static uint get(uint a) {
        return (a >> OFFSET) & DigitMask<BITS>::value;
    }
};
template<int BITS, int OFFSET> struct ExtractMaskedValue<uint2, BITS, OFFSET> {
    __device__
    inline static uint2 get(const uint2& a) {
        return make_uint2((a.x >> OFFSET) & DigitMask<BITS>::value,
                          (a.y >> OFFSET) & DigitMask<BITS>::value);
    }
};
template<int BITS, int OFFSET> struct ExtractMaskedValue<uint4, BITS, OFFSET> {
    __device__
    inline static uint4 get(const uint4& a) {
        return make_uint4((a.x >> OFFSET) & DigitMask<BITS>::value,
                          (a.y >> OFFSET) & DigitMask<BITS>::value,
                          (a.z >> OFFSET) & DigitMask<BITS>::value,
                          (a.w >> OFFSET) & DigitMask<BITS>::value);
    }
};


template <int NUM_ELEMENTS>
__device__ inline
int SerialReduce(uint segment[]) {
    uint reduce = segment[0];
    #pragma unroll
    for (int i = 1; i < NUM_ELEMENTS; ++i) {
        reduce += segment[i];
    }
    return reduce;
}

template <int NUM_ELEMENTS>
__device__ inline
void SerialScan(uint segment[]) {
    uint sum = 0;
    #pragma unroll
    for (int i = 0; i < NUM_ELEMENTS; ++i){
        uint x = segment[i];
        segment[i] = sum;
        sum += x;
    }
}


/*
 * NUM_ELEMENTS = 64  =>  Active Threads = 32
 * NUM_ELEMENTS = 32  =>  Active Threads = 16
 * NUM_ELEMENTS = 16  =>  Active Threads = 8
 * NUM_ELEMENTS = 8   =>  Active Threads = 4
 * NUM_ELEMENTS = 4   =>  Active Threads = 2
 * NUM_ELEMENTS = 2   =>  Active Threads = 1
 */
template <int NUM_ELEMENTS>
__device__ inline
void WarpReduce(volatile uint* shared_storage, int tid) {
    if (tid < (NUM_ELEMENTS >> 1)) {
        if (NUM_ELEMENTS > 32) shared_storage[tid] += shared_storage[tid + 32];
        if (NUM_ELEMENTS > 16) shared_storage[tid] += shared_storage[tid + 16];
        if (NUM_ELEMENTS >  8) shared_storage[tid] += shared_storage[tid +  8];
        if (NUM_ELEMENTS >  4) shared_storage[tid] += shared_storage[tid +  4];
        if (NUM_ELEMENTS >  2) shared_storage[tid] += shared_storage[tid +  2];
        if (NUM_ELEMENTS >  1) shared_storage[tid] += shared_storage[tid +  1];
    }
}

__device__ inline
void WarpScanExclusive(volatile uint* shared_storage, int tid) {
    uint x = shared_storage[tid];
    int sum = x;
    if (tid >= 1)  sum += shared_storage[tid - 1];
    shared_storage[tid] = sum;
    if (tid >= 2)  sum += shared_storage[tid - 2];
    shared_storage[tid] = sum;
    if (tid >= 4)  sum += shared_storage[tid - 4];
    shared_storage[tid] = sum;
    if (tid >= 8)  sum += shared_storage[tid - 8];
    shared_storage[tid] = sum;
    if (tid >= 16) sum += shared_storage[tid - 16];
    shared_storage[tid] = sum;
    shared_storage[tid] = sum - x;
}


__device__ inline
void WarpScanExclusive(volatile uint* shared_storage, const uint* const global_in, int tid) {
    uint x = global_in[tid];
    shared_storage[tid] = x;
    int sum = x;
    if (tid >= 1)  sum += shared_storage[tid - 1];
    shared_storage[tid] = sum;
    if (tid >= 2)  sum += shared_storage[tid - 2];
    shared_storage[tid] = sum;
    if (tid >= 4)  sum += shared_storage[tid - 4];
    shared_storage[tid] = sum;
    if (tid >= 8)  sum += shared_storage[tid - 8];
    shared_storage[tid] = sum;
    if (tid >= 16) sum += shared_storage[tid - 16];
    shared_storage[tid] = sum;
    shared_storage[tid] = sum - x;
}

__device__ inline
void WarpScanInclusive(volatile uint* shared_storage, int tid) {
    uint x = shared_storage[tid];
    int sum = x;
    if (tid >= 1)  sum += shared_storage[tid - 1];
    shared_storage[tid] = sum;
    if (tid >= 2)  sum += shared_storage[tid - 2];
    shared_storage[tid] = sum;
    if (tid >= 4)  sum += shared_storage[tid - 4];
    shared_storage[tid] = sum;
    if (tid >= 8)  sum += shared_storage[tid - 8];
    shared_storage[tid] = sum;
    if (tid >= 16) sum += shared_storage[tid - 16];
    shared_storage[tid] = sum;
}

__device__ inline
void WarpScanInclusive(volatile uint* shared_storage, const uint* const global_in, int tid) {
    uint x = global_in[tid];
    shared_storage[tid] = x;
    int sum = x;
    if (tid >= 1)  sum += shared_storage[tid - 1];
    shared_storage[tid] = sum;
    if (tid >= 2)  sum += shared_storage[tid - 2];
    shared_storage[tid] = sum;
    if (tid >= 4)  sum += shared_storage[tid - 4];
    shared_storage[tid] = sum;
    if (tid >= 8)  sum += shared_storage[tid - 8];
    shared_storage[tid] = sum;
    if (tid >= 16) sum += shared_storage[tid - 16];
    shared_storage[tid] = sum;
}



__device__
void ReduceCTA(volatile uint* smem, int CTA_SIZE) {

    const int tid = threadIdx.x;

    if (CTA_SIZE >= 512) { if (tid < 256) { smem[tid] += smem[tid + 256]; } __syncthreads(); }
    if (CTA_SIZE >= 256) { if (tid < 128) { smem[tid] += smem[tid + 128]; } __syncthreads(); }
    if (CTA_SIZE >= 128) { if (tid <  64) { smem[tid] += smem[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        volatile uint* smemp = smem;
        if (CTA_SIZE >= 64) smemp[tid] += smemp[tid + 32];
        if (CTA_SIZE >= 32) smemp[tid] += smemp[tid + 16];
        if (CTA_SIZE >= 16) smemp[tid] += smemp[tid +  8];
        if (CTA_SIZE >=  8) smemp[tid] += smemp[tid +  4];
        if (CTA_SIZE >=  4) smemp[tid] += smemp[tid +  2];
        if (CTA_SIZE >=  2) smemp[tid] += smemp[tid +  1];
    }
}

// Original version see: http://www.moderngpu.com/intro/scan.html
__device__
void ScanCTA(volatile int* array, volatile uint* localSum, int CTA_SIZE)
{
    __shared__ volatile int scan_storage[NUM_WARPS * SCAN_STRIDE];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = (WARP_SIZE - 1) & tid;

    volatile int* s = scan_storage + SCAN_STRIDE * warp + lane + WARP_SIZE / 2;
    s[-16] = 0;

    // Read from global memory to shared memory
    int x = array[tid];
    s[0] = x;

    // Run inclusive scan on each warp's data
    int sum = x;
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        int offset = 1 << i;
        sum += s[-offset];
        s[0] = sum;
    }

    // Synchronize to make all the totals available to the reduction code
    __syncthreads();

    __shared__ volatile int totals[NUM_WARPS + NUM_WARPS / 2];

    if (tid < NUM_WARPS) {

        int total = scan_storage[SCAN_STRIDE * tid + WARP_SIZE / 2 + WARP_SIZE - 1];

        totals[tid] = 0;
        volatile int* s2 = totals + NUM_WARPS / 2 + tid;
        int totalsSum = total;
        s2[0] = total;

        #pragma unroll
        for (int i = 0; i < LOG_NUM_WARPS; ++i) {
            int offset = 1 << i;
            totalsSum += s2[-offset];
            s2[0] = totalsSum;
        }

        totals[tid] = totalsSum - total;
    }

    // Synchronize to make the block scan available to all warps
    __syncthreads();

    // Add the block scan to the inclusive sum of the block
    sum += totals[warp];

    // Write the inclusive and excusive scans to global memory
    // inclusive scan elements
    int a = sum;

    // exclusive scan elements
    array[tid] = sum - x;

    if (tid == CTA_SIZE - 1) {
        localSum[0] = a;
    }
    __syncthreads();
}



template<typename Key, int PASS, int RADIX_DIGITS, int BITS>
__global__
void UpsweepReduceKernel(
        uint* d_spine,
        Key* d_inputKeys,
        uint num_tiles,
        uint num_elements)
{

    const Key* tile = d_inputKeys + num_tiles * blockDim.x * blockIdx.x;
    const int tid = threadIdx.x;

    uint histogram[RADIX_DIGITS];

    #pragma unroll
    for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
        histogram[radix] = 0;
    }

    Key element;
    int tileCounter = 0;

    // Process all tiles and update local histogram
    while (num_tiles > tileCounter) {

        element = tile[tid];               // load element from input array tile

        element = ExtractMaskedValue<Key, BITS, PASS * BITS>::get(element);
        UpdateHistogram<Key>::apply(histogram, element);

        tile += blockDim.x;
        tileCounter += 1;
    }

    /*
     * Now load local histograms to shared memory
     * and reduce them cooperatively
     */
    __shared__ uint reduce_storage[RADIX_DIGITS][NUM_THREADS];

    // Load local histograms to shared memory
    #pragma unroll
    for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
        reduce_storage[radix][tid] = histogram[radix];
    }
    __syncthreads();

    // Reduce cooperatively
    #pragma unroll
    for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
        ReduceCTA(&reduce_storage[radix][0], blockDim.x);
    }

    // Store results to global buffer
    if (tid < RADIX_DIGITS) {
        uint accum = reduce_storage[tid][0];
        d_spine[blockIdx.x + tid * CTAs] = accum; // scattered store
        d_spine[tid + RADIX_DIGITS * CTAs] = 0;   // scattered store
    }
}

template<int RADIX_DIGITS>
__global__
void SpineKernel(uint* d_spine, size_t num_elements)
{
    const uint tid = threadIdx.x;

    uint* const totalsPtr = d_spine + CTAs * RADIX_DIGITS;

    volatile __shared__ uint accumSum[RADIX_DIGITS];

    if (tid < RADIX_DIGITS)
        accumSum[tid] = 0;
    __syncthreads();

    if (tid < CTAs)
    {
        #pragma unroll
        for (int radix = 0; radix < RADIX_DIGITS; ++radix)
        {
            uint* const scanPtr = d_spine + CTAs * radix;
            ScanCTA((int*)scanPtr, &accumSum[radix], CTAs);
        }
    }

    __shared__ uint padd[1];
    if (0 == tid) padd[0] = 0;

    if (tid < RADIX_DIGITS)
    {
        ScanCTA((int*)accumSum, padd, RADIX_DIGITS);
        totalsPtr[tid] = accumSum[tid];
    }
}



template<typename Key, int PASS, int RADIX_DIGITS, int BITS>
__device__
inline void BallotScanCTAILP(
        Key* globalOutput,
        Key* globalTile,
        volatile uint* countTotals,
        const uint* scannedSum,
        const uint* totalsSum)
{

    const uint tid  = threadIdx.x;
    const uint lane = (WARP_SIZE - 1) & tid;
    const uint warp = tid >> 5;

    Key val = globalTile[tid];

    Key u = ExtractMaskedValue<Key, BITS, PASS * BITS>::get(val);

    // Ballot scan tha flags in warp
    uint bits[RADIX_DIGITS];
    #pragma unroll
    for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
        bits[radix] = __ballot_sync(FULL_MASK, (radix == u) ? 1 : 0);
    }

    //uint mask = bfi(0, 0xFFFFFFFF, 0, lane); // bfi doesn't compile!
    //uint mask = (1 << lane) - 1;

    // how many before tid voted true
    uint excl[RADIX_DIGITS];
    #pragma unroll
    for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
        excl[radix] = __popc(((1 << lane) - 1) & bits[radix]);
    }

    // how many from all warp voted true
    uint warpTotal[RADIX_DIGITS];
    #pragma unroll
    for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
        warpTotal[radix] = __popc(bits[radix]);
    }

    // Store each warp total into shared memory
    __shared__ volatile uint shared[RADIX_DIGITS][NUM_WARPS];
    if (0 == lane)
    {
        #pragma unroll
        for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
            shared[radix][warp] = warpTotal[radix];
        }
    }

    // Inclusive scan the warp totals
    __syncthreads();
    if (tid < NUM_WARPS)
    {
        uint x[RADIX_DIGITS];

        #pragma unroll
        for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
            x[radix] = shared[radix][tid];
        }

        #pragma unroll
        for (int i = 0; i < LOG_NUM_WARPS; ++i)
        {
            uint offset = 1 << i;
            if (tid >= offset)
            {
                #pragma unroll
                for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
                    x[radix] += shared[radix][tid - offset];
                }
            }

            #pragma unroll
            for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
                shared[radix][tid] = x[radix];
            }
        }
    }
    __syncthreads();

    // Add the exclusive scanned warp totals into excl
    uint blockTotal[RADIX_DIGITS];

    #pragma unroll
    for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
        blockTotal[radix] = shared[radix][NUM_WARPS - 1];
    }

    #pragma unroll
    for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
        excl[radix] += shared[radix][warp] - warpTotal[radix];
    }

    uint ctotals[RADIX_DIGITS];

    #pragma unroll
    for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
        ctotals[radix] = countTotals[radix];
    }
    __syncthreads();

    #pragma unroll
    for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
        excl[radix] += ctotals[radix];
    }


    #pragma unroll
    for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
        if (radix == u) {
            uint index = excl[radix] + scannedSum[radix] + totalsSum[radix];
            globalOutput[index] = val;
        }
    }

    __syncthreads();

    if (0 == tid) {
        #pragma unroll
        for (int radix = 0; radix < RADIX_DIGITS; ++radix) {
            countTotals[radix] = blockTotal[radix] + ctotals[radix];
        }
    }
    __syncthreads();
}


template<typename Key, int PASS, int RADIX_DIGITS, int BITS>
__global__
void DownsweepScanKernel(
        Key*  d_outputKeys,
        Key*  d_inputKeys,
        const uint* const spine,
        uint  num_tiles,
        uint  num_elements)
{
    const int tid = threadIdx.x;
    Key* tile = d_inputKeys + num_tiles * blockDim.x * blockIdx.x;

    volatile __shared__ uint countTotals[RADIX_DIGITS];   // changes for each tile
    __shared__ uint scannedSum[RADIX_DIGITS]; // constant in whole CTA for each tile
    __shared__ uint totalsSum[RADIX_DIGITS];  // constant in whole CTA for each tile

    if (tid < RADIX_DIGITS) {
        countTotals[tid] = 0;                                // zero before traverse tiles
        scannedSum[tid]  = spine[blockIdx.x + CTAs * tid]; // load offsets
        totalsSum[tid]   = spine[CTAs * RADIX_DIGITS + tid];      // load scan of total sums
    }
    __syncthreads();


    int tileCounter = 0;

    while (num_tiles > tileCounter++) {

        BallotScanCTAILP<Key, PASS, RADIX_DIGITS, BITS>(
                d_outputKeys,
                tile,
                countTotals,
                scannedSum,
                totalsSum);

        tile += blockDim.x;
    }
}

#endif /* XRADIXSORTKERNEL_H_ */
