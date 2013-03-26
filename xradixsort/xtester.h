/*
 * xtester.h
 *
 *  Created on: Mar 23, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef XTESTER_H_
#define XTESTER_H_

#include <cstdio>
#include <vector>
#include <algorithm>

#include "xconfig.h"
#include "xdevice.h"

template <typename Key>
class RadixSortTester
{
public:

    std::vector<Key> sample;
    std::vector<Key> result;
    size_t           array_size;
    size_t           sample_max;
    bool             compare;

    RadixSortTester(int argc, char** argv, const CudaDevice& device, bool cmp = true)
    : array_size(0)
    , sample_max(0)
    , compare(cmp)
    , currentDevice(device) {

        int tilesPerCTA = 0;
        if (argc == 3) {
            tilesPerCTA = atoi(argv[1]);
            array_size = CTAs * NUM_THREADS * tilesPerCTA;
            sample_max = atoi(argv[2]);
        } else if (argc == 2) {
            tilesPerCTA = atoi(argv[1]);
            array_size = CTAs * NUM_THREADS * tilesPerCTA;
            sample_max = 1024;
        } else {
            tilesPerCTA = 4;
            array_size = CTAs * NUM_THREADS * tilesPerCTA;
            sample_max = 1024;
        }
        sample.resize(array_size);
        result.resize(array_size);
        printf("C:    %d\nB:    %d\n", CTAs, tilesPerCTA);
    }

    ~RadixSortTester() {
    }

    void InitSample() {
        std::srand(time(0));
        for (size_t s(0); s < sample.size(); ++s) {
            sample[s] = static_cast<Key>(std::rand() % sample_max);
        }
        printf("N:    %d\n", sample.size());
        printf("SAMPLE_MAX: %d\n", array_size);
    }

    void ShowStats(float elapsedTimeMs) {
        printf("Computation time:       "
                "%f [ms]\n", elapsedTimeMs);
        printf("Peak bandwidth:         "
                "%.3f   [GB/s]\n", currentDevice.peakBandwidth);
    }

    void CompareResults() {
        if (!compare) return;
        std::sort(sample.begin(), sample.end());
        bool incorrect(false);
        size_t s(0);
        for (; s < sample.size(); ++s) {
            if (result[s] == sample[s]) continue;
            incorrect = true;
            break;
        }
        if (incorrect) {
            printf("Incorrect at %d:  Ref: %d vs. Dev: %d\n", s, sample[s], result[s]);
        } else {
            printf("Perfectly correct!\n");
        }
    }

private:
    const CudaDevice& currentDevice;
    RadixSortTester(const RadixSortTester& r);
    RadixSortTester& operator=(const RadixSortTester& r);
};

#endif /* XTESTER_H_ */
