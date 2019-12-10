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

void rowOperations(const shared_ptr< double[]>& a, const shared_ptr<double []>& b, const int maxIdx, const int mSize)
{
    double tmp = a[maxIdx]/b[maxIdx];
    for(int i=0; i<mSize; i++)
    {
        a[i] = a[i] - b[i]*tmp;
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
    }

    shared_ptr < pair < shared_ptr < double[] > , int >[]> readStorage(new pair<shared_ptr<double[] >, int > [mSize]);

    auto startTime = chrono::high_resolution_clock::now();

    queue<int> tasks;
    tasks.push(0);
    int started = 0;
    int completed = 0;

    countThreads = 8;
    #pragma omp parallel num_threads(countThreads)
    {
        int workRow = -1;
        while (true)
        {
            bool out = false;
            #pragma omp critical
            {
                if ( !tasks.empty() ) {
                    workRow = tasks.front();
                    omp_set_lock(&rowReadLock[workRow]);
                    tasks.pop();
                    started++;
                } else if (completed == mSize) {
                    out = true;
                } else {
                    workRow = -1;
                }
            };
            if (out) {
                break;
            }
            if(workRow == -1) {
                usleep(10);
                continue;
            }

            int maxIdx = getVectorMax(matrix[workRow], mSize);

            if (workRow + 1 < mSize)
            {
                int i = workRow + 1;
                omp_set_lock(&rowReadLock[i]);
                rowOperations(matrix[i], matrix[workRow], maxIdx, mSize + 1);
                omp_unset_lock(&rowReadLock[i]);
                if (i == workRow + 1)
                    tasks.push(i);
            }

            queue<int> localTasks;
            for (int i = workRow + 2; i < mSize; i++)
            {
                if ( omp_test_lock(&rowReadLock[i]) )
                {
                    rowOperations(matrix[i], matrix[workRow], maxIdx, mSize + 1);
                    omp_unset_lock(&rowReadLock[i]);
                }
            }

            while (!localTasks.empty())
            {
                int i = localTasks.front();
                localTasks.pop();
                omp_set_lock(&rowReadLock[i]);
                rowOperations(matrix[i], matrix[workRow], maxIdx, mSize + 1);
                omp_unset_lock(&rowReadLock[i]);
            }

            #pragma omp atomic
            completed++;
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
    cout <<"Время выполнения " << chrono::duration_cast<chrono::duration<double>>(endTime - startTime).count() << endl;
    //fstream statFile("gauss_stat.txt", ios::out | ios::app);
    //statFile << countThreads << "," <<  chrono::duration_cast<chrono::duration<double>>(endTime - startTime).count() <<endl;
    return 0;
}
