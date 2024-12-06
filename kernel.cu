
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <stdlib.h> 
#include <stdio.h>
#include <fstream>
using namespace std;

/*Map Class, generates initial map array, position array, and direction array. Takes size of pen and spawn probability as constructor arguments*/
class Map {
public:
    //Map array, index is the grid position (intersection number), stored value is occupancy (0= empty, 1= full, >1= overfull, negative= exit)
    //Size is equal to the pen length * pen width, plus the border of the pen.
    int* M;

    //Position array, holds the current possition of all goats. Index is goat number, stored value is grid position (same as index for map array).
    //Size is equal to pen length * pen width, but some indexes might not be used. Position (-1) is used for goats outside of pen.
    int* goatPos;

    //Direction array, holds the direction each goat is facing. Index is goat number, and corresponds to the same goat as in Position array. Stored value is direction.
    //(0= up, 1= right, 2= right, 3= left). Size is same as Position array
    int* goatDir;

    //Map Generation
    Map(int size, int probability) {
        //Allocate memory for arrays
        M = (int*)malloc((size+2) * (size+2) * sizeof(int));
        goatPos = (int*)malloc(size * size * sizeof(int));
        goatDir = (int*)malloc(size * size * sizeof(int));
        for (int i = 0; i < (size + 2); i++) {
            //Mark top and bottom boundaries as occupied
            M[i * (size + 2)] = 1;
            M[(i * (size + 2)) + (size + 1)] = 1;
        }
        for (int i = 1; i <= size; i++) {
            //Mark side boundaries as occupied
            M[i] = 1;
            M[((size + 2) * (size + 1)) + i] = 1;
        }
        //Create exit at the middle of the bottom boundary
        M[(size + 2) / 2] = -5;

        //variable for goat index
        int count = 0;

        //Instantiate goat positions
        for (int i = 1; i <= size; i++) {
            for (int j = 1; j <= size; j++) {
                //Use spawn probability to instantiate goats
                if ((rand() % 100) >= probability) {
                    //if random exceeds probability value, set the current map position to empty
                    M[(i * (size + 2)) + j] = 0;
                }
                else {
                    //if random is within probability value, fill the current map position, set goat postion to current map position, then increment goat index
                    M[(i * (size + 2)) + j] = 1;
                    goatPos[count] = (i * (size + 2)) + j;
                    goatDir[count] = 0;
                    count++;
                }
            }
        }

        //Set remaining goat positions to (-1), so that the gpu knows to ignore them
        for (int i = count; i < (size * size); i++) {
            goatPos[i] = -1;
            goatDir[i] = -1;
        }
    }

    //Destructor, Free Memory
    ~Map() {
        free(M);
        free(goatPos);
        free(goatDir);
    }
};


/*Goat movement function, executed by gpu threads. Takes the 3 info arrays, plus the pen size and a random number*/
__global__ void moveGoat(int *goatPos, int *goatDir, int *M, int size, int r) {
    //change size from internal pen size to total pen size
    size += 2;
    int i = threadIdx.x;

    //ignore goats outside of pen
    if (goatPos[i] < 0) {
        return;
    }

    int pastPos = goatPos[i];

    //Random chance for goat change its direction (turn right)
    goatDir[i] = (goatDir[i] + r%2) % 4;

    //Try to move in direction the goat is facing
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

    //If current goat position is overfull, backtrack
    if (M[goatPos[i]] > 1) {
        M[goatPos[i]]--;
        goatPos[i] = pastPos;
        M[goatPos[i]]++;
    }

    //If current goat position is on exit, remove goat from map
    if (M[goatPos[i]] < 0) {
        M[goatPos[i]]--;
        goatPos[i] = -1;
        goatDir[i] = -1;
    }
}

int main()
{
    int user = 0;

    //
    //
    //
    //
    int penSize = 11;                          //Set length and width of pen (internal size, does not include borders)
    int probability = 55;                      //Set % probability a goat will be instatiated at any position in the map, integer between 0 and 100
    //
    //
    //
    //

    //Create arrays
    Map x(penSize, probability);

    //Print map (upside down, becuase terminal prints from top to bottom)
    for (int i = 0; i < (penSize+2); i++) {
        for (int j = 0; j < (penSize+2); j++) {
            printf("%d", x.M[(i * (penSize+2)) + j]);
        }
        printf("\n");
    }

    ofstream file("output.txt");

    int* M_gpu, * goatPos_gpu, * goatDir_gpu;

    //allocate gpu memory for map, position, and direction arrays
    cudaMalloc((void**) &M_gpu, (penSize + 2) * (penSize + 2) * sizeof(int));
    cudaMalloc((void**) &goatPos_gpu, penSize * penSize * sizeof(int));
    cudaMalloc((void**) &goatDir_gpu, penSize * penSize * sizeof(int));

    //copy arrays from cpu to gpu memory
    cudaMemcpy(M_gpu, x.M, (penSize + 2) * (penSize + 2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(goatPos_gpu, x.goatPos, penSize * penSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(goatDir_gpu, x.goatDir, penSize * penSize * sizeof(int), cudaMemcpyHostToDevice);

    //cycle count for debuging
    int count = 0;

    while (user == 0) {

        //random number for movement function
        int r = (rand() % 100);

        //move goats
        moveGoat << <1, (penSize * penSize) >> > (goatPos_gpu, goatDir_gpu, M_gpu, penSize, r);
        cudaDeviceSynchronize;

        //copy goat positions to cpu array so it can be output
        cudaMemcpy(x.goatPos, goatPos_gpu, penSize * penSize * sizeof(int), cudaMemcpyDeviceToHost);

        user = 1;

        //output goat positions to output file in form (x, y)
        //Each column is a goat, each row is a movement cycle
        for (int i = 0; i < (penSize * penSize); i++) {
            file << "(" << (x.goatPos[i] % (penSize + 2)) << ", " << (x.goatPos[i] / (penSize + 2)) << ") ";
            if ((x.goatPos[i] % (penSize + 2)) < 10 && (x.goatPos[i] % (penSize + 2)) != -1) {
                file << " ";
            }
            if ((x.goatPos[i] / (penSize + 2)) < 10) {
                file << " ";
            }

            //Continue while loop unless all goats are out of pen
            if (x.goatPos[i] != -1) {
                user = 0;
            }
        }
        file << endl;


        //extra debug function for collision detection

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


        //extra debug funcion that prints map each cycle, and waits for user input to start next cycle

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

    //close file and free gpu memory
    file.close();
    cudaFree(M_gpu);
    cudaFree(goatPos_gpu);
    cudaFree(goatDir_gpu);
    return 0;
}