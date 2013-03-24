/*
 * xconfig.h
 *
 *  Created on: Mar 23, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef XCONFIG_H_
#define XCONFIG_H_

#define __ENABLE_KERNEL_SYNCHRONIZATION__ 1

//#define CTAs 64
//
//#define WARP_SIZE 32
//#define NUM_THREADS 256
//#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
//#define LOG_NUM_THREADS 8
//#define LOG_NUM_WARPS (LOG_NUM_THREADS - 5)
//#define SCAN_STRIDE (WARP_SIZE + WARP_SIZE / 2 + 1)

#define CTAs 128

#define WARP_SIZE 32
#define NUM_THREADS 128
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_THREADS 7
#define LOG_NUM_WARPS (LOG_NUM_THREADS - 5)
#define SCAN_STRIDE (WARP_SIZE + WARP_SIZE / 2 + 1)

#endif /* XCONFIG_H_ */
