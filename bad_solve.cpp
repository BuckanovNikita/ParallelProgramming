#include <iostream>
#include <omp.h>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <vector>
#include <memory>

using namespace std;

int getVectorMax(const shared_ptr<double[]>& vec, const int mSize)
{
    double res = 0;
    int idx = 0;
    for(int i=0; i<mSize; i++)
        if ( abs(res) < abs(vec[i]))
        {
            idx = i;
            res = vec[i];
        }
    return idx;
}

int lowBound(const int threadIdx, const int threadPoolSize, const int mSize)
{
    return mSize/threadPoolSize * threadIdx;
}

int upperBound(const int threadIdx, const int threadPoolSize, const int mSize)
{
    if (threadIdx + 1 == threadPoolSize)
        return mSize;
    return mSize/threadPoolSize * (threadIdx + 1);
}

bool isOwner(const int rowIdx, const int threadIdx, const int threadPoolSize, const int mSize)
{
    return ( (lowBound(threadIdx, threadPoolSize, mSize) <= rowIdx)  && (rowIdx < upperBound(threadIdx, threadPoolSize, mSize)) );
}

void rowOperations(const shared_ptr< double[]>& a, const shared_ptr<double []>& b, const int maxIdx, const int mSize)
{
    double tmp = a[maxIdx]/b[maxIdx];
    for(int i=0; i<mSize; i++)
    {
        a[i] = a[i] - b[i]*tmp;
    }
}

void ownerOperations(const shared_ptr<shared_ptr< double[]>[]>& matrix, const shared_ptr < omp_lock_t[] >& rowReadLock,
                     int startIdx, int endIdx,const shared_ptr<int[]>& maxValues, int mSize)
{
    maxValues[startIdx] = getVectorMax(matrix[startIdx], mSize);
    omp_unset_lock(&rowReadLock[startIdx]);
    for(int i=startIdx + 1; i < endIdx; i++)
    {
        for(int j = startIdx; j<i; j++) {
            rowOperations(matrix[i], matrix[j], maxValues[j], mSize);
        }
        maxValues[i] = getVectorMax(matrix[i], mSize);
        omp_unset_lock(&rowReadLock[i]);
    }
}


int main(int argc, char* argv[]) {

    if (argc < 2) {
        cerr << "Using app {count_of_threads}" << endl;
        return 0;
    }

    int countThreads = atoi(argv[1]);
    if (countThreads < 1) {
        cerr << "numThreads must be number more than 0" << endl;
    }

    fstream input("gaussInput.txt", std::fstream::in);
    int mSize = 0;
    input >> mSize;
    if (mSize < 2) {
        cerr << "Size of matrix less then 2" << endl;
        return 0;
    }


    //vector<vector<double>> matrix(mSize);
    shared_ptr < shared_ptr<double[]>[] > matrix(new shared_ptr<double[]>[mSize]);
    vector<double> x(mSize);

    for (int i = 0; i < mSize; i++) {
        matrix[i] = shared_ptr<double[]>(new double[mSize + 1]);
        for (int j = 0; j < mSize; j++) {
            input >> matrix[i][j];
        }
    }

    for (int i = 0; i < mSize; i++)
        input >> x[i];

    for (int i = 0; i < mSize; i++)
        input >> matrix[i][mSize];

    shared_ptr < omp_lock_t[] > rowReadLock(new omp_lock_t[mSize]);
    for (int i = 0; i < mSize; i++) {
        omp_init_lock(&rowReadLock[i]);
        omp_set_lock(&rowReadLock[i]);
    }

    shared_ptr < pair < shared_ptr < double[] > , int >[]> readStorage(new pair<shared_ptr<double[] >, int > [mSize]);

    shared_ptr<int[]> maxValues(new int [mSize]);

    auto startTime = chrono::high_resolution_clock::now();
    countThreads = 4;
#pragma omp parallel num_threads(countThreads) //shared (rowReadLock, matrix, readStorage)
    {
        int threadIdx = omp_get_thread_num();

        for (int i = 0; i < mSize; i++) {

            int maxIdx = getVectorMax(matrix[i], mSize);
            if (isOwner(i, threadIdx, countThreads, mSize)) {
                ownerOperations(matrix, rowReadLock,
                                lowBound(threadIdx, countThreads, mSize),
                                upperBound(threadIdx, countThreads, mSize), maxValues, mSize);
                break;
            }

            omp_set_lock(&rowReadLock[i]);
            omp_unset_lock(&rowReadLock[i]);

            for (int rowIdx = max(lowBound(threadIdx, countThreads, mSize), i + 1);
                 rowIdx < upperBound(threadIdx, countThreads, mSize); rowIdx++) {
                rowOperations(matrix[rowIdx], matrix[i], maxValues[i], mSize+1);
                if (i + 1 == rowIdx)
                    omp_unset_lock(&rowReadLock[i+1]);
            }

        }
    }
    cout << setprecision(4) << endl;

    vector<double> answer(mSize);
    double maxDelta = 0;
    for(int i = mSize-1; i > -1; i--)
    {
        int maxIdx = getVectorMax(matrix[i], mSize);
        for(int j = 0; j<mSize; j++)
            matrix[i][mSize] = matrix[i][mSize] - answer[j] * matrix[i][j];
        answer[maxIdx] = matrix[i][mSize] / matrix[i][maxIdx];
        maxDelta = max(maxDelta, abs(answer[maxIdx] - x[maxIdx]));
    }
    auto endTime = chrono::high_resolution_clock::now();
    cout <<"Погрешность вычислений " << maxDelta <<endl;
    cout <<"Время выполнения " << chrono::duration_cast<chrono::duration<double>>(endTime - startTime).count() <<endl ;
    return 0;
}
