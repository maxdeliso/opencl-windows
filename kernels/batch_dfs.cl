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
