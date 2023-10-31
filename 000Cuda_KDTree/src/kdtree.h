// kdtree.h
#pragma once

#ifndef KDTREE_H
#define KDTREE_H

//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
//#include <iostream>
//
//#include <curand_kernel.h>



struct Point3D {
    float coords[3];
};

struct KdNode {
    int axis;
    float value;
    int left;
    int right;
};


#endif
