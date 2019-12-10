#include <iostream>
#include <omp.h>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <vector>
#include <memory>
#include <queue>
#include <unistd.h>
#include <sys/time.h>

using namespace std;


int main(int argc, char* argv[]) {

    if (argc < 2) {
        cerr << "Using app {count_of_threads}" << endl;
        return 0;
    }

    int countThreads = atoi(argv[1]);
    if (countThreads < 1) {
        cerr << "numThreads must be number more than 0" << endl;
    }


    fstream input("matmulInput.txt", ios::in);
    int mSize = 0;
    input >> mSize;
    shared_ptr<double[]> matrixA(new double[mSize*mSize]);
    shared_ptr<double[]> matrixB(new double[mSize*mSize]);
    shared_ptr<double[]> matrixC(new double[mSize*mSize]);
    shared_ptr<double[]> matrixC_(new double[mSize*mSize]);
    for(int i=0; i<mSize*mSize; i++)
    {
        input >> matrixA[i];
    }
    for(int i=0; i<mSize*mSize; i++)
    {
        input >> matrixB[i%mSize * mSize + i/mSize];
    }
    for(int i=0; i<mSize*mSize; i++)
    {
        input >> matrixC_[i];
    }

    auto startTime = chrono::high_resolution_clock::now();

    countThreads = 12;
    #pragma omp parallel for num_threads(countThreads)
    for(int i=0; i<mSize; i++)
    {
        int iOff = i * mSize;
        for(int j=0; j<mSize; j++)
        {
            int jOff = j * mSize;
            double tot = 0;
            for(int k=0; k<mSize; k++)
            {
                tot += matrixA[iOff + k] * matrixB[jOff + k];
            }
            matrixC[i*mSize + j] = tot;
        }
    }
    auto endTime = chrono::high_resolution_clock::now();
    cout << "Время выполнения: " << chrono::duration_cast<chrono::duration<double>>(endTime - startTime).count() <<endl;
    double delta = 0;
    for(int i=0; i<mSize*mSize; i++)
    {
        delta = max( delta, abs(matrixC[i] - matrixC_[i]));
    }
    cout << "Погрешность вычислений: " << delta <<endl;

    fstream statFile("matmul_stat.txt", ios::out | ios::app);
    statFile << countThreads << "," <<  chrono::duration_cast<chrono::duration<double>>(endTime - startTime).count() <<endl;
}
