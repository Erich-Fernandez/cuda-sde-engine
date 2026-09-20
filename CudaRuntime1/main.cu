#include "GBM_kernel.h"




int main() {
    GpuContext ctx;
    
    printf("test run :\n");
    GBM(1, 0, 1, 100, T, 1000000, ctx, 1);
    printf("---------end of test run line---------\n\n");

    printf("\n\nNAIVE GBM COMPLETE PARALLEL SUM\n\n");
    for (int i = 0; i < 10; i++) {
        GBM(1, 0, 1, 160, T, 100000000, ctx, 1);
    }

    printf("\n\nSUPREMUM GBM COMPLETE PARALLEL SUM\n\n");
    for (int i = 0; i < 10; i++) {
        int steps_per_path = 16 * (i + 1);
        GBM_sup(1, 0, 1, 160, T, 100000000, ctx, 1);
    }

    printf("\n\nNAIVE GBM ONLY REDUCTION PARALLEL SUM\n\n");
    for (int i = 0; i < 10; i++) {
        int steps_per_path = 16 * (i + 1);
        GBM_upgradedParallel(1, 0, 1, 160, T, 100000000, ctx, 1);
    }
}