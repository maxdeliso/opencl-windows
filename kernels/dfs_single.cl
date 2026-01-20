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