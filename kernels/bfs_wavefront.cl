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
