
#include <cuda_runtime.h>

#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../../shared/include/utility.h"

//#include "kernel.cu"
#include "kdtree.h"
//#include <curand_kernel.h>
//#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>



extern "C" void launchBuildKDTree(Point3D * points, int numPoints, KdNode * tree);
extern "C" void launchNearestNeighborSearch(Point3D * d_points, KdNode * d_tree, Point3D * d_query, Point3D * d_result, int numPoints, int numThreadsPerBlock);

// Define a Point3D structure

//const int numPoints = 1000000; // Adjust this to your needs
const float minValue = 0.0f; // Minimum coordinate value
const float maxValue = 100.0f; // Maximum coordinate value


// Function to generate random points and populate the given array
void generateRandomPoints(Point3D* d_points, unsigned int numPoints) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(minValue, maxValue);

    for (unsigned int i = 0; i < numPoints; ++i) {
        d_points[i].coords[0] = dis(gen); // X-coordinate
        d_points[i].coords[1] = dis(gen); // Y-coordinate
        d_points[i].coords[2] = dis(gen); // Z-coordinate
    }
}

//const int threadsPerBlock = 256;



int main() {
    // Initialize CUDA and allocate memory for point cloud and K-D Tree on the GPU
    cudaSetDevice(0);
    int numPoints = 1000; // Adjust the number of points as needed
    Point3D* cpu_points = (Point3D*)malloc(sizeof(Point3D) * numPoints);
    Point3D* gpu_points;
    KdNode* d_tree;

    cudaMalloc((void**)&gpu_points, sizeof(Point3D) * numPoints);

    cudaMalloc((void**)&d_tree, sizeof(KdNode) * numPoints);

    // Generate random points for the K-D Tree
    generateRandomPoints(cpu_points, numPoints);
    cudaMemcpy(gpu_points, cpu_points, sizeof(Point3D) * numPoints, cudaMemcpyHostToDevice);

    launchBuildKDTree(gpu_points, numPoints, d_tree);


	Point3D* d_query;
	Point3D* d_result;
	Point3D query = { 0.5f, 0.5f, 0.5f }; // Adjust the query point as needed
	cudaMalloc((void**)&d_query, sizeof(Point3D));
	cudaMalloc((void**)&d_result, sizeof(Point3D));

	cudaMemcpy(d_query, &query, sizeof(Point3D), cudaMemcpyHostToDevice);

	// Launch the nearest neighbor search
    launchNearestNeighborSearch(cpu_points, d_tree, d_query, d_result, numPoints, 256);

	// Copy the result back to the host
	Point3D result;
	cudaMemcpy(&result, d_result, sizeof(Point3D), cudaMemcpyDeviceToHost);

	std::cout << "Nearest Neighbor: (" << result.coords[0] << ", " << result.coords[1] << ", " << result.coords[2] << ")\n";

     //Don't forget to free the allocated memory when done
    delete cpu_points;
    cudaFree(gpu_points);
    cudaFree(d_tree);
    cudaFree(d_query);
    cudaFree(d_result);



    return 0;
}