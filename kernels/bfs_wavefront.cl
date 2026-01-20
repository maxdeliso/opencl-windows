kernel void bfs_wavefront(global const int* offsets,
                          global const int* edges,
                          global atomic_int* levels,
                          global int* frontierActive,
                          int currentLevel,
                          int nodeCount)
{
    int tid = get_global_id(0);
    if (tid >= nodeCount)
    {
        return;
    }

    if (atomic_load(&levels[tid]) == currentLevel)
    {
        int edgeStart = offsets[tid];
        int edgeEnd = offsets[tid + 1];
        for (int edge = edgeStart; edge < edgeEnd; ++edge)
        {
            int neighbor = edges[edge];
            int expected = -1;
            bool success = atomic_compare_exchange_strong(&levels[neighbor], &expected, currentLevel + 1);
            if (success)
            {
                *frontierActive = 1;
            }
        }
    }
}
