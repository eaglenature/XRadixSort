/*
 * xradixsort.cu
 *
 *  Created on: Mar 23, 2013
 *      Author: eaglenature@gmail.com
 */

#include "xtestrunner.h"

#include "xradixsort.h"
#include "xtester.h"
#include "xdevice.h"
#include "xtimer.h"


CUDATEST(PrimarySortingTest, 0)
{
    typedef uint KeyType;

    CudaDevice device;
    device.Init();

    RadixSortTester<KeyType> tester(0, 0, device);
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


#if 0
int main(int argc, char** argv)
{
    TestRunner::GetInstance().RunAll();
}
#else
int main(int argc, char** argv)
{
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
#endif
