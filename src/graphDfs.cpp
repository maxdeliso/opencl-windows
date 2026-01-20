/*
 * Graph DFS demo helpers.
 */
#include "graphDfs.h"

#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <queue>

namespace
{
cl_device_id SelectPreferredDevice(const cl_device_id* devices, cl_uint deviceCount)
{
    cl_device_id selected = devices[0];
    for (cl_uint i = 0; i < deviceCount; ++i)
    {
        cl_device_type type = 0;
        if (clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type), &type, NULL) == CL_SUCCESS)
        {
            if (type & CL_DEVICE_TYPE_GPU)
            {
                selected = devices[i];
                break;
            }
        }
    }

    return selected;
}

double DurationMs(const std::chrono::high_resolution_clock::time_point& start,
                  const std::chrono::high_resolution_clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device, cl_int* err)
{
#if defined(CL_VERSION_2_0)
    const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    return clCreateCommandQueueWithProperties(context, device, props, err);
#else
    return clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, err);
#endif
}
} // namespace

GraphCsr GenerateRandomGraph(int nodeCount, int extraEdgesPerNode, unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, nodeCount - 1);

    std::vector<std::vector<int>> adjacency(static_cast<size_t>(nodeCount));

    for (int node = 0; node < nodeCount; ++node)
    {
        adjacency[node].push_back((node + 1) % nodeCount);
        for (int e = 0; e < extraEdgesPerNode; ++e)
        {
            adjacency[node].push_back(dist(rng));
        }
    }

    GraphCsr graph;
    graph.nodeCount = nodeCount;
    graph.offsets.resize(static_cast<size_t>(nodeCount + 1), 0);
    size_t totalEdges = 0;
    for (int node = 0; node < nodeCount; ++node)
    {
        graph.offsets[static_cast<size_t>(node)] = static_cast<int>(totalEdges);
        totalEdges += adjacency[node].size();
    }
    graph.offsets[static_cast<size_t>(nodeCount)] = static_cast<int>(totalEdges);
    graph.edges.reserve(totalEdges);
    for (int node = 0; node < nodeCount; ++node)
    {
        for (int neighbor : adjacency[node])
        {
            graph.edges.push_back(neighbor);
        }
    }

    return graph;
}

std::vector<int> DfsCpu(const GraphCsr& graph, int startNode)
{
    std::vector<unsigned char> visited(static_cast<size_t>(graph.nodeCount), 0);
    std::vector<int> stack;
    std::vector<int> order;
    stack.reserve(static_cast<size_t>(graph.nodeCount));
    order.reserve(static_cast<size_t>(graph.nodeCount));

    visited[static_cast<size_t>(startNode)] = 1;
    stack.push_back(startNode);
    while (!stack.empty())
    {
        int node = stack.back();
        stack.pop_back();

        order.push_back(node);

        int edgeStart = graph.offsets[static_cast<size_t>(node)];
        int edgeEnd = graph.offsets[static_cast<size_t>(node + 1)];
        for (int edge = edgeEnd - 1; edge >= edgeStart; --edge)
        {
            int neighbor = graph.edges[static_cast<size_t>(edge)];
            if (!visited[static_cast<size_t>(neighbor)])
            {
                visited[static_cast<size_t>(neighbor)] = 1;
                stack.push_back(neighbor);
            }
        }
    }

    return order;
}

std::vector<int> BfsCpuLevels(const GraphCsr& graph, int startNode)
{
    std::vector<int> levels(static_cast<size_t>(graph.nodeCount), -1);
    std::queue<int> q;
    levels[static_cast<size_t>(startNode)] = 0;
    q.push(startNode);

    while (!q.empty())
    {
        int node = q.front();
        q.pop();
        int edgeStart = graph.offsets[static_cast<size_t>(node)];
        int edgeEnd = graph.offsets[static_cast<size_t>(node + 1)];
        int nextLevel = levels[static_cast<size_t>(node)] + 1;
        for (int edge = edgeStart; edge < edgeEnd; ++edge)
        {
            int neighbor = graph.edges[static_cast<size_t>(edge)];
            if (levels[static_cast<size_t>(neighbor)] == -1)
            {
                levels[static_cast<size_t>(neighbor)] = nextLevel;
                q.push(neighbor);
            }
        }
    }

    return levels;
}

bool RunGraphDfsDemo(cl_context context, const cl_device_id* devices, cl_uint deviceCount)
{
    const int kNodeCount = 200000;
    const int kExtraEdgesPerNode = 6;
    const unsigned int kSeed = 1337;

    shrLog("\nGraph DFS Demo:\n\n");
    shrLog(" Generating random graph with cycles (%d nodes, %d+1 edges per node)...\n",
           kNodeCount, kExtraEdgesPerNode);
    shrLog(" GPU DFS runs a single work-item to preserve DFS order.\n");
    GraphCsr graph = GenerateRandomGraph(kNodeCount, kExtraEdgesPerNode, kSeed);

    auto cpuStart = std::chrono::high_resolution_clock::now();
    std::vector<int> cpuOrder = DfsCpu(graph, 0);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = DurationMs(cpuStart, cpuEnd);

    cl_device_id device = SelectPreferredDevice(devices, deviceCount);
    if (device == NULL)
    {
        shrLog(" No suitable OpenCL device found for DFS demo.\n");
        return false;
    }

    cl_int err = CL_SUCCESS;
    cl_command_queue queue = CreateCommandQueue(context, device, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating command queue.\n", err);
        return false;
    }

    const char* programSource = R"CLC(
kernel void dfs_single(global const int* offsets,
                       global const int* edges,
                       int nodeCount,
                       global int* outOrder,
                       global int* outCount,
                       global int* stack,
                       global char* visited)
{
    int sp = 0;
    int outIdx = 0;
    visited[0] = 1;
    stack[sp++] = 0;

    while (sp > 0)
    {
        int node = stack[--sp];
        outOrder[outIdx++] = node;

        int edgeStart = offsets[node];
        int edgeEnd = offsets[node + 1];
        for (int edge = edgeEnd - 1; edge >= edgeStart; --edge)
        {
            int neighbor = edges[edge];
            if (!visited[neighbor])
            {
                visited[neighbor] = 1;
                stack[sp++] = neighbor;
            }
        }
    }

    outCount[0] = outIdx;
}
)CLC";
    size_t programLength = std::strlen(programSource);
    const char* sourceList[] = { programSource };
    cl_program program = clCreateProgramWithSource(context, 1, sourceList, &programLength, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating OpenCL program.\n", err);
        clReleaseCommandQueue(queue);
        return false;
    }

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i building OpenCL program.\n", err);
        oclLogBuildInfo(program, device);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_kernel kernel = clCreateKernel(program, "dfs_single", &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating kernel.\n", err);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    const size_t offsetsBytes = graph.offsets.size() * sizeof(int);
    const size_t edgesBytes = graph.edges.size() * sizeof(int);
    const size_t orderBytes = static_cast<size_t>(graph.nodeCount) * sizeof(int);
    const size_t visitedBytes = static_cast<size_t>(graph.nodeCount) * sizeof(unsigned char);

    cl_mem offsetsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          offsetsBytes, graph.offsets.data(), &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating offsets buffer.\n", err);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_mem edgesBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        edgesBytes, graph.edges.data(), &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating edges buffer.\n", err);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_mem orderBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, orderBytes, NULL, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating order buffer.\n", err);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_mem countBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating count buffer.\n", err);
        clReleaseMemObject(orderBuffer);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_mem stackBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, orderBytes, NULL, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating stack buffer.\n", err);
        clReleaseMemObject(countBuffer);
        clReleaseMemObject(orderBuffer);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_mem visitedBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, visitedBytes, NULL, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating visited buffer.\n", err);
        clReleaseMemObject(stackBuffer);
        clReleaseMemObject(countBuffer);
        clReleaseMemObject(orderBuffer);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    const int zeroInt = 0;
    const unsigned char zeroChar = 0;
    err = clEnqueueFillBuffer(queue, countBuffer, &zeroInt, sizeof(zeroInt), 0, sizeof(int), 0, NULL, NULL);
    err |= clEnqueueFillBuffer(queue, visitedBuffer, &zeroChar, sizeof(zeroChar), 0, visitedBytes, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i initializing GPU buffers.\n", err);
        clReleaseMemObject(visitedBuffer);
        clReleaseMemObject(stackBuffer);
        clReleaseMemObject(countBuffer);
        clReleaseMemObject(orderBuffer);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &offsetsBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &edgesBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &graph.nodeCount);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &orderBuffer);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &countBuffer);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &stackBuffer);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &visitedBuffer);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i setting kernel arguments.\n", err);
        clReleaseMemObject(visitedBuffer);
        clReleaseMemObject(stackBuffer);
        clReleaseMemObject(countBuffer);
        clReleaseMemObject(orderBuffer);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    size_t globalWorkSize[1] = { 1 };
    cl_event kernelEvent = NULL;

    auto gpuTotalStart = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &kernelEvent);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i launching kernel.\n", err);
        clReleaseMemObject(visitedBuffer);
        clReleaseMemObject(stackBuffer);
        clReleaseMemObject(countBuffer);
        clReleaseMemObject(orderBuffer);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    clWaitForEvents(1, &kernelEvent);

    cl_ulong kernelStart = 0;
    cl_ulong kernelEnd = 0;
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(kernelStart), &kernelStart, NULL);
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(kernelEnd), &kernelEnd, NULL);
    clReleaseEvent(kernelEvent);
    double gpuKernelMs = static_cast<double>(kernelEnd - kernelStart) * 1.0e-6;

    int gpuCount = 0;
    std::vector<int> gpuOrder(static_cast<size_t>(graph.nodeCount), 0);
    err = clEnqueueReadBuffer(queue, countBuffer, CL_TRUE, 0, sizeof(int), &gpuCount, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, orderBuffer, CL_TRUE, 0,
                               static_cast<size_t>(gpuCount) * sizeof(int),
                               gpuOrder.data(), 0, NULL, NULL);
    auto gpuTotalEnd = std::chrono::high_resolution_clock::now();
    double gpuTotalMs = DurationMs(gpuTotalStart, gpuTotalEnd);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i reading GPU results.\n", err);
        clReleaseMemObject(visitedBuffer);
        clReleaseMemObject(stackBuffer);
        clReleaseMemObject(countBuffer);
        clReleaseMemObject(orderBuffer);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    bool match = (gpuCount == static_cast<int>(cpuOrder.size()));
    if (match)
    {
        for (int i = 0; i < gpuCount; ++i)
        {
            if (gpuOrder[static_cast<size_t>(i)] != cpuOrder[static_cast<size_t>(i)])
            {
                match = false;
                break;
            }
        }
    }

    shrLog(" CPU DFS visited %u nodes in %.3f ms.\n", static_cast<unsigned int>(cpuOrder.size()), cpuMs);
    shrLog(" GPU DFS visited %u nodes in %.3f ms (kernel), %.3f ms (total).\n",
           static_cast<unsigned int>(gpuCount), gpuKernelMs, gpuTotalMs);
    shrLog(" DFS order comparison: %s\n", match ? "MATCH" : "MISMATCH");

    clReleaseMemObject(visitedBuffer);
    clReleaseMemObject(stackBuffer);
    clReleaseMemObject(countBuffer);
    clReleaseMemObject(orderBuffer);
    clReleaseMemObject(edgesBuffer);
    clReleaseMemObject(offsetsBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);

    return match;
}

bool RunGraphBatchDfsDemo(cl_context context, const cl_device_id* devices, cl_uint deviceCount)
{
    const int kNodesPerGraph = 32;
    const int kNumGraphs = 16384;
    const int kMatrixSize = kNodesPerGraph * kNodesPerGraph;
    const int kEdgeChancePercent = 30;
    const unsigned int kSeed = 7331;

    shrLog("\nGraph Batch DFS Demo:\n\n");
    shrLog(" Generating %d graphs (%d nodes each)...\n", kNumGraphs, kNodesPerGraph);

    std::mt19937 rng(kSeed);
    std::uniform_int_distribution<int> dist(0, 99);

    std::vector<int> allMatrices(static_cast<size_t>(kNumGraphs) * kMatrixSize);
    std::vector<int> cpuResults(static_cast<size_t>(kNumGraphs), 0);
    std::vector<int> gpuResults(static_cast<size_t>(kNumGraphs), 0);

    for (size_t i = 0; i < allMatrices.size(); ++i)
    {
        allMatrices[i] = (dist(rng) < kEdgeChancePercent) ? 1 : 0;
    }

    shrLog(" Running CPU serial DFS across all graphs...\n");
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int graph = 0; graph < kNumGraphs; ++graph)
    {
        const int base = graph * kMatrixSize;
        std::vector<int> stack;
        stack.reserve(static_cast<size_t>(kNodesPerGraph));
        unsigned int visited = 1u;
        int count = 0;

        stack.push_back(0);
        while (!stack.empty())
        {
            int current = stack.back();
            stack.pop_back();
            ++count;

            const int rowStart = base + current * kNodesPerGraph;
            for (int node = 0; node < kNodesPerGraph; ++node)
            {
                if (allMatrices[static_cast<size_t>(rowStart + node)] == 1)
                {
                    if (((visited >> node) & 1u) == 0)
                    {
                        visited |= (1u << node);
                        stack.push_back(node);
                    }
                }
            }
        }

        cpuResults[static_cast<size_t>(graph)] = count;
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = DurationMs(cpuStart, cpuEnd);

    cl_device_id device = SelectPreferredDevice(devices, deviceCount);
    if (device == NULL)
    {
        shrLog(" No suitable OpenCL device found for batch DFS demo.\n");
        return false;
    }

    cl_int err = CL_SUCCESS;
    cl_command_queue queue = CreateCommandQueue(context, device, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating command queue.\n", err);
        return false;
    }

    const char* programSource = R"CLC(
#define NODES 32
#define MAX_STACK 32
kernel void batch_dfs(global const int* adj_matrices,
                      global int* results)
{
    int graphId = get_global_id(0);
    int offset = graphId * NODES * NODES;

    int stack[MAX_STACK];
    int stackTop = 0;
    stack[0] = 0;

    unsigned int visited = 1u;
    int count = 0;

    while (stackTop >= 0)
    {
        int current = stack[stackTop--];
        ++count;

        int rowStart = offset + current * NODES;
        for (int node = 0; node < NODES; ++node)
        {
            if (adj_matrices[rowStart + node] == 1)
            {
                if (((visited >> node) & 1u) == 0)
                {
                    if (stackTop < MAX_STACK - 1)
                    {
                        visited |= (1u << node);
                        stack[++stackTop] = node;
                    }
                }
            }
        }
    }

    results[graphId] = count;
}
)CLC";

    size_t programLength = std::strlen(programSource);
    const char* sourceList[] = { programSource };
    cl_program program = clCreateProgramWithSource(context, 1, sourceList, &programLength, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating OpenCL program.\n", err);
        clReleaseCommandQueue(queue);
        return false;
    }

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i building OpenCL program.\n", err);
        oclLogBuildInfo(program, device);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_kernel kernel = clCreateKernel(program, "batch_dfs", &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating kernel.\n", err);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_mem matrixBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(int) * allMatrices.size(), allMatrices.data(), &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating matrix buffer.\n", err);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_mem resultsBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                          sizeof(int) * kNumGraphs, NULL, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating results buffer.\n", err);
        clReleaseMemObject(matrixBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrixBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &resultsBuffer);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i setting kernel arguments.\n", err);
        clReleaseMemObject(resultsBuffer);
        clReleaseMemObject(matrixBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    size_t globalWorkSize[1] = { static_cast<size_t>(kNumGraphs) };
    auto gpuStart = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i launching batch kernel.\n", err);
        clReleaseMemObject(resultsBuffer);
        clReleaseMemObject(matrixBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }
    clFinish(queue);

    err = clEnqueueReadBuffer(queue, resultsBuffer, CL_TRUE, 0,
                              sizeof(int) * kNumGraphs, gpuResults.data(), 0, NULL, NULL);
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    double gpuMs = DurationMs(gpuStart, gpuEnd);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i reading batch results.\n", err);
        clReleaseMemObject(resultsBuffer);
        clReleaseMemObject(matrixBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    int mismatches = 0;
    for (int i = 0; i < kNumGraphs; ++i)
    {
        if (cpuResults[static_cast<size_t>(i)] != gpuResults[static_cast<size_t>(i)])
        {
            ++mismatches;
        }
    }

    shrLog(" CPU batch DFS time: %.3f ms.\n", cpuMs);
    shrLog(" GPU batch DFS time: %.3f ms.\n", gpuMs);
    if (gpuMs > 0.0)
    {
        shrLog(" Speedup: %.2fx\n", cpuMs / gpuMs);
    }
    shrLog(" Verification: %s (%d mismatches)\n",
           mismatches == 0 ? "SUCCESS" : "FAILED", mismatches);

    clReleaseMemObject(resultsBuffer);
    clReleaseMemObject(matrixBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);

    return mismatches == 0;
}

bool RunGraphWavefrontBfsDemo(cl_context context, const cl_device_id* devices, cl_uint deviceCount)
{
    const int kNodeCount = 200000;
    const int kExtraEdgesPerNode = 6;
    const unsigned int kSeed = 4242;

    shrLog("\nGraph Wavefront BFS Demo:\n\n");
    shrLog(" Generating CSR graph (%d nodes, %d+1 edges per node)...\n",
           kNodeCount, kExtraEdgesPerNode);
    shrLog(" GPU BFS uses a topology-driven wavefront (thread per node).\n");

    GraphCsr graph = GenerateRandomGraph(kNodeCount, kExtraEdgesPerNode, kSeed);

    shrLog(" Running CPU queue BFS...\n");
    auto cpuStart = std::chrono::high_resolution_clock::now();
    // BFS distance (level) from source node; -1 means unreachable.
    std::vector<int> cpuLevels = BfsCpuLevels(graph, 0);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = DurationMs(cpuStart, cpuEnd);

    cl_device_id device = SelectPreferredDevice(devices, deviceCount);
    if (device == NULL)
    {
        shrLog(" No suitable OpenCL device found for wavefront BFS demo.\n");
        return false;
    }

    cl_int err = CL_SUCCESS;
    cl_command_queue queue = CreateCommandQueue(context, device, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating command queue.\n", err);
        return false;
    }

    const char* programSource = R"CLC(
kernel void bfs_wavefront(global const int* offsets,
                          global const int* edges,
                          global int* levels,
                          global int* frontierActive,
                          int currentLevel,
                          int nodeCount)
{
    int tid = get_global_id(0);
    if (tid >= nodeCount)
    {
        return;
    }

    if (levels[tid] == currentLevel)
    {
        int edgeStart = offsets[tid];
        int edgeEnd = offsets[tid + 1];
        for (int edge = edgeStart; edge < edgeEnd; ++edge)
        {
            int neighbor = edges[edge];
            int oldVal = atomic_cmpxchg(&levels[neighbor], -1, currentLevel + 1);
            if (oldVal == -1)
            {
                *frontierActive = 1;
            }
        }
    }
}
)CLC";

    size_t programLength = std::strlen(programSource);
    const char* sourceList[] = { programSource };
    cl_program program = clCreateProgramWithSource(context, 1, sourceList, &programLength, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating OpenCL program.\n", err);
        clReleaseCommandQueue(queue);
        return false;
    }

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i building OpenCL program.\n", err);
        oclLogBuildInfo(program, device);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_kernel kernel = clCreateKernel(program, "bfs_wavefront", &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating kernel.\n", err);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    const size_t offsetsBytes = graph.offsets.size() * sizeof(int);
    const size_t edgesBytes = graph.edges.size() * sizeof(int);
    const size_t levelsBytes = static_cast<size_t>(graph.nodeCount) * sizeof(int);

    // GPU levels mirror the BFS distance array for validation.
    std::vector<int> gpuLevels(static_cast<size_t>(graph.nodeCount), -1);
    gpuLevels[0] = 0;

    cl_mem offsetsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          offsetsBytes, graph.offsets.data(), &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating offsets buffer.\n", err);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_mem edgesBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        edgesBytes, graph.edges.data(), &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating edges buffer.\n", err);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_mem levelsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         levelsBytes, gpuLevels.data(), &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating levels buffer.\n", err);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    cl_mem activeBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i creating active flag buffer.\n", err);
        clReleaseMemObject(levelsBuffer);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &offsetsBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &edgesBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &levelsBuffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &activeBuffer);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i setting kernel arguments.\n", err);
        clReleaseMemObject(activeBuffer);
        clReleaseMemObject(levelsBuffer);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    shrLog(" Running GPU wavefront BFS...\n");
    size_t globalWorkSize[1] = { static_cast<size_t>(graph.nodeCount) };
    int currentLevel = 0;
    int activeFlag = 1;
    int iterations = 0;
    const int maxIterations = graph.nodeCount;

    auto gpuStart = std::chrono::high_resolution_clock::now();
    while (activeFlag && iterations < maxIterations)
    {
        activeFlag = 0;
        err = clEnqueueWriteBuffer(queue, activeBuffer, CL_TRUE, 0, sizeof(int), &activeFlag, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            shrLog(" Error %i updating active flag.\n", err);
            break;
        }

        err = clSetKernelArg(kernel, 4, sizeof(int), &currentLevel);
        err |= clSetKernelArg(kernel, 5, sizeof(int), &graph.nodeCount);
        if (err != CL_SUCCESS)
        {
            shrLog(" Error %i setting wavefront args.\n", err);
            break;
        }

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            shrLog(" Error %i launching wavefront kernel.\n", err);
            break;
        }

        err = clEnqueueReadBuffer(queue, activeBuffer, CL_TRUE, 0, sizeof(int), &activeFlag, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            shrLog(" Error %i reading active flag.\n", err);
            break;
        }

        if (activeFlag)
        {
            ++currentLevel;
        }
        ++iterations;
    }
    clFinish(queue);
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    double gpuMs = DurationMs(gpuStart, gpuEnd);

    err = clEnqueueReadBuffer(queue, levelsBuffer, CL_TRUE, 0, levelsBytes, gpuLevels.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        shrLog(" Error %i reading BFS levels.\n", err);
        clReleaseMemObject(activeBuffer);
        clReleaseMemObject(levelsBuffer);
        clReleaseMemObject(edgesBuffer);
        clReleaseMemObject(offsetsBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        return false;
    }

    int mismatches = 0;
    for (int i = 0; i < graph.nodeCount; ++i)
    {
        if (cpuLevels[static_cast<size_t>(i)] != gpuLevels[static_cast<size_t>(i)])
        {
            ++mismatches;
        }
    }

    shrLog(" CPU BFS time: %.3f ms.\n", cpuMs);
    shrLog(" GPU BFS time: %.3f ms (%d iterations).\n", gpuMs, iterations);
    if (gpuMs > 0.0)
    {
        shrLog(" Speedup: %.2fx\n", cpuMs / gpuMs);
    }
    shrLog(" Validation: %s (%d mismatches)\n",
           mismatches == 0 ? "SUCCESS" : "FAILED", mismatches);

    clReleaseMemObject(activeBuffer);
    clReleaseMemObject(levelsBuffer);
    clReleaseMemObject(edgesBuffer);
    clReleaseMemObject(offsetsBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);

    return mismatches == 0;
}
