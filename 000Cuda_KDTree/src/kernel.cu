
#include <cuda_runtime.h>

//#include <cuda_runtime_api.h>
#include <iostream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
//#include "../../shared/include/utility.h"


#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include "kdtree.h"


namespace cg = cooperative_groups;


// Find the median along the specified axis
__device__ float findMedian(Point3D* points, int start, int end, int axis) {
    int n = end - start;
    // Sort the points along the specified axis
    for (int i = start; i < end - 1; i++) {
        for (int j = i + 1; j < end; j++) {
            if (points[i].coords[axis] > points[j].coords[axis]) {
                // Swap points
                Point3D temp = points[i];
                points[i] = points[j];
                points[j] = temp;
            }
        }
    }
    // Calculate the median
    int medianIdx = start + n / 2;
    return points[medianIdx].coords[axis];
}

// Partition the points based on the median along the specified axis
__device__ int partition(Point3D* points, int start, int end, int axis, float median) {
    int left = start;
    int right = end - 1;

    while (left <= right) {
        while (points[left].coords[axis] < median) {
            left++;
        }

        while (points[right].coords[axis] > median) {
            right--;
        }

        if (left <= right) {
            // Swap points
            Point3D temp = points[left];
            points[left] = points[right];
            points[right] = temp;
            left++;
            right--;
        }
    }

    return left - start;
}


// Define a comparator function for sorting points along the chosen axis
bool comparePointsByAxis(const Point3D& a, const Point3D& b, int axis) {
    return a.coords[axis] < b.coords[axis];
}

// Define a WorkItem structure to represent subproblems
struct WorkItem {
    int start;
    int end;
    int depth;
    int left;
    int right;
};

// Define a Worklist data structure
struct Worklist {
    WorkItem* items;
    int capacity;
    int size;
    __device__ Worklist() {
		this->items = nullptr;
		this->capacity = 0;
		this->size = 0;
	}
    __device__ Worklist(WorkItem* items, int capacity) {
        this->items = items;
        this->capacity = capacity;
        this->size = 0;
    }

    __device__ void push(int start, int end, int depth, int left, int right) {
        if (size < capacity) {
            WorkItem item;
            item.start = start;
            item.end = end;
            item.depth = depth;
            item.left = left;
            item.right = right;
            items[size] = item;
            size++;
        }
    }

    __device__ WorkItem pop() {
        if (size > 0) {
            size--;
            return items[size];
        }
        else {
            WorkItem emptyItem = { 0, 0, 0, 0, 0 };
            return emptyItem;
        }
    }

    __device__ bool isEmpty() {
        return size == 0;
    }
};


__global__ void buildKDTree(Point3D* points, KdNode* tree, int numPoints) {

    Worklist worklist; // Create a worklist
    worklist.push(0, numPoints, 0, 0, 0); // Start with the entire point range and initial depth

    while (!worklist.isEmpty()) {
        WorkItem subproblem = worklist.pop();
        int start = subproblem.start;
        int end = subproblem.end;
        int depth = subproblem.depth;

        if (end - start <= 0) {
            continue; // Subproblem is empty, move to the next
        }

        // Your KD-Tree construction logic here

        int axis = depth % 3; // Choose splitting axis
        float median = findMedian(points, start, end, axis);
        int leftCount = partition(points, start, end, axis, median);

        tree[start].axis = axis;
        tree[start].value = median;
        tree[start].left = start + 1;
        tree[start].right = start + 1 + leftCount;

        // Push left and right child subproblems onto the worklist
        worklist.push(start + 1, start + 1 + leftCount, depth + 1, subproblem.left, subproblem.right);
        worklist.push(start + 1 + leftCount, end, depth + 1, subproblem.right, subproblem.left);
    }
}


__global__ void initRandomStates(curandState* states, int numThreads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numThreads) {
        curand_init(0, tid, 0, &states[tid]);
    }
}
extern "C" void launchBuildKDTree(Point3D * points, int numPoints, KdNode * tree) {
    dim3 gridSize(1); // Set your grid size
    dim3 blockSize(256); // Set your block size
    curandState* states;

    // Allocate and initialize random states (you may need to adjust the size)
    cudaMalloc((void**)&states, sizeof(curandState) * numPoints);
    initRandomStates <<<gridSize, blockSize>>> (states, numPoints);

    // Launch the KD-Tree building kernel via the wrapper function
    buildKDTree <<<gridSize, blockSize>>> (points, tree, numPoints);

    // Free random states (if necessary)
    cudaFree(states);
}





__global__ void nearestNeighborSearch(Point3D* points, KdNode* tree, Point3D query, Point3D* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize variables to keep track of the best neighbor and the best distance
    float bestDistance = FLT_MAX;
    Point3D bestNeighbor = { 0.0f, 0.0f, 0.0f };

    // Define a cooperative grid and thread group
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int stack[64]; // Stack for tree traversal (adjust size if necessary)
    int stackPtr = 0; // Stack pointer

    int currentNode = 0; // Start from the root node
    int currentAxis = 0; // Axis to compare for splitting
    float currentDistance = 0.0f;

    while (true) {
        // Traverse the KD-Tree iteratively
        if (currentNode != -1) {
            // Calculate the distance from the query point to the current node
            currentDistance = powf(query.coords[0] - points[currentNode].coords[0], 2) +
                powf(query.coords[1] - points[currentNode].coords[1], 2) +
                powf(query.coords[2] - points[currentNode].coords[2], 2);

            // Update the best neighbor if necessary
            if (currentDistance < bestDistance) {
                bestDistance = currentDistance;
                bestNeighbor = points[currentNode];
            }

            // Determine which child node to visit next
            int nextNode = -1;
            if (query.coords[currentAxis] < points[currentNode].coords[currentAxis]) {
                nextNode = tree[currentNode].left;
            }
            else {
                nextNode = tree[currentNode].right;
            }

            // Calculate the next axis for splitting
            currentAxis = (currentAxis + 1) % 3;

            // Push the other child onto the stack for later traversal
            if (nextNode != -1 && currentDistance < bestDistance) {
                stack[stackPtr++] = (currentAxis + 1) % 3; // Push the other axis
                stack[stackPtr++] = nextNode; // Push the other child node
            }

            currentNode = nextNode;
        }
        else {
            // Backtrack to the parent node
            if (stackPtr > 0) {
                currentAxis = stack[--stackPtr];
                currentNode = stack[--stackPtr];
            }
            else {
                break; // The stack is empty, and we're done
            }
        }

        // Synchronize to ensure all threads in the block have completed their work
        block.sync();
    }

    // Store the best neighbor in the result
    result[tid] = bestNeighbor;
}



extern "C" void launchNearestNeighborSearch(Point3D * d_points, KdNode * d_tree, Point3D * d_query, Point3D * d_result, int numPoints, int numThreadsPerBlock) {
    // Calculate the number of blocks based on the number of points and threads per block
    int numBlocks = (numPoints + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // Launch the kernel
    nearestNeighborSearch <<<numBlocks, numThreadsPerBlock >>> (d_points, d_tree, *d_query, d_result);
    cudaDeviceSynchronize(); // Wait for the kernel to finish
}
