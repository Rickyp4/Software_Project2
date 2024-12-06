
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <stdlib.h> 
#include <stdio.h>
#include <fstream>
using namespace std;

class Map {
public:
    int* M;
    int* goatPos;
    int* goatDir;
    Map(int size, int probability) {
        M = (int*)malloc((size+2) * (size+2) * sizeof(int));
        goatPos = (int*)malloc(size * size * sizeof(int));
        goatDir = (int*)malloc(size * size * sizeof(int));
        for (int i = 0; i < (size + 2); i++) {
            M[i * (size + 2)] = 1;
            M[(i * (size + 2)) + (size + 1)] = 1;
        }
        for (int i = 1; i <= size; i++) {
            M[i] = 1;
            M[((size + 2) * (size + 1)) + i] = 1;
        }
        M[(size + 2) / 2] = -5;

        int count = 0;

        for (int i = 1; i <= size; i++) {
            for (int j = 1; j <= size; j++) {
                if ((rand() % 100) >= probability) {
                    M[(i * (size + 2)) + j] = 0;
                }
                else {
                    M[(i * (size + 2)) + j] = 1;
                    goatPos[count] = (i * (size + 2)) + j;
                    goatDir[count] = 0;
                    count++;
                }
            }
        }
        for (int i = count; i < (size * size); i++) {
            goatPos[i] = -1;
            goatDir[i] = -1;
        }
    }
    ~Map() {
        free(M);
        free(goatPos);
        free(goatDir);
    }
};


__global__ void moveGoat(int *goatPos, int *goatDir, int *M, int size, int r) {
    size += 2;
    int i = threadIdx.x;
    if (goatPos[i] < 0) {
        return;
    }
    int pastPos = goatPos[i];
    goatDir[i] = (goatDir[i] + r%2) % 4;
    if (goatDir[i] == 0) {
        if (M[goatPos[i] + size] < 1) {
            M[goatPos[i]]--;
            goatPos[i] += size;
            M[goatPos[i]]++;
        }
    }
    if (goatDir[i] == 1) {
        if (M[goatPos[i] + 1] < 1) {
            M[goatPos[i]]--;
            goatPos[i] += 1;
            M[goatPos[i]]++;
        }
    }
    if (goatDir[i] == 2) {
        if (M[goatPos[i] - size] < 1) {
            M[goatPos[i]]--;
            goatPos[i] -= size;
            M[goatPos[i]]++;
        }
    }
    if (goatDir[i] == 3) {
        if (M[goatPos[i] - 1] < 1) {
            M[goatPos[i]]--;
            goatPos[i] -= 1;
            M[goatPos[i]]++;
        }
    }
    if (M[goatPos[i]] > 1) {
        M[goatPos[i]]--;
        goatPos[i] = pastPos;
        M[goatPos[i]]++;
    }
    if (M[goatPos[i]] < 0) {
        M[goatPos[i]]--;
        goatPos[i] = -1;
        goatDir[i] = -1;
    }
}

int main()
{
    int user = 0;
    int penSize = 18;
    int probability = 50;
    Map x(penSize, probability);
    for (int i = 0; i < (penSize+2); i++) {
        for (int j = 0; j < (penSize+2); j++) {
            printf("%d", x.M[(i * (penSize+2)) + j]);
        }
        printf("\n");
    }

    ofstream file("output.txt");

    int* M_gpu, * goatPos_gpu, * goatDir_gpu;

    cudaMalloc((void**) &M_gpu, (penSize + 2) * (penSize + 2) * sizeof(int));
    cudaMalloc((void**) &goatPos_gpu, penSize * penSize * sizeof(int));
    cudaMalloc((void**) &goatDir_gpu, penSize * penSize * sizeof(int));

    cudaMemcpy(M_gpu, x.M, (penSize + 2) * (penSize + 2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(goatPos_gpu, x.goatPos, penSize * penSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(goatDir_gpu, x.goatDir, penSize * penSize * sizeof(int), cudaMemcpyHostToDevice);


    int count = 0;

    while (user == 0) {
        /*
        cudaMemcpy(M_gpu, x.M, (penSize + 2) * (penSize + 2) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(goatPos_gpu, x.goatPos, penSize * penSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(goatDir_gpu, x.goatDir, penSize * penSize * sizeof(int), cudaMemcpyHostToDevice);
        */

        int r = (rand() % 100);

        moveGoat << <1, (penSize * penSize) >> > (goatPos_gpu, goatDir_gpu, M_gpu, penSize, r);
        cudaDeviceSynchronize;

        cudaMemcpy(x.goatPos, goatPos_gpu, penSize * penSize * sizeof(int), cudaMemcpyDeviceToHost);

        user = 1;

        for (int i = 0; i < (penSize * penSize); i++) {
            file << "(" << (x.goatPos[i] % (penSize + 2)) << ", " << (x.goatPos[i] / (penSize + 2)) << ") ";
            if ((x.goatPos[i] % (penSize + 2)) < 10 && (x.goatPos[i] % (penSize + 2)) != -1) {
                file << " ";
            }
            if ((x.goatPos[i] / (penSize + 2)) < 10) {
                file << " ";
            }
            if (x.goatPos[i] != -1) {
                user = 0;
            }
        }
        file << endl;

        /*
        cudaMemcpy(x.M, M_gpu, (penSize + 2) * (penSize + 2) * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < ((penSize + 2) * (penSize + 2)); i++) {
            if (x.M[i] > 1) {
                cout << count << "  broken (" << (i % (penSize + 2)) << ", " << (i / (penSize + 2)) << ") \n";
                //break;
                for (int k = 0; k < (penSize + 2); k++) {
                    for (int j = 0; j < (penSize + 2); j++) {
                        printf("%d", x.M[(k * (penSize + 2)) + j]);
                    }
                    printf("\n");
                }
            }
        }
        //*/
        /*

        for (int i = 0; i < (penSize + 2); i++) {
            for (int j = 0; j < (penSize + 2); j++) {
                printf("%d", x.M[(i * (penSize + 2)) + j]);
            }
            printf("\n");
        }
        printf("\n");
        cin >> user;
        //*/
        count++;
    }
    file.close();
    cudaFree(M_gpu);
    cudaFree(goatPos_gpu);
    cudaFree(goatDir_gpu);
    return 0;
}