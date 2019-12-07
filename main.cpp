#include <iostream>
#include <omp.h>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <fstream>

using namespace std;

inline double intFunction(const double x)
{
    return sqrt(1 - x*x);
}

inline double gridPoint(const int i, const int N)
{
    return (double)i/N;
}



int main(int argc, char* argv[]) {
    if(argc < 3)
    {
        cerr << "Using app {count_of_points} {count_of_threads}" <<endl;
        return 0;
    }

    int gridSize =  atoi(argv[1]);
    int countThreads = atoi(argv[2]);
    if (gridSize < 1 || countThreads < 1)
    {
        cerr << "gridSize and numThreads must be number more than 0" <<endl;
    }

    double sum = 0;

    omp_set_num_threads(countThreads);
    auto startTime = chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum)
    for(int i=0; i<gridSize; i++) {
        {
            cout << omp_get_thread_num() << " " <<  i << endl;
            cout.flush();
            #pragma omp atomic
            sum += intFunction(gridPoint(i, gridSize))*gridPoint(1, gridSize);
        };

    }
    auto endTime = chrono::high_resolution_clock::now();
    chrono::duration<double> execTime = chrono::duration_cast<chrono::duration<double>>(endTime - startTime);
    cout << setprecision(10) <<endl;
    cout << "Calculated value of Pi=" << 4 * sum << " delta=" << 4 * sum - M_PI << endl;
    cout << "Exec time " << execTime.count() <<endl;

    //fstream statFile;
    //statFile.open("stat.txt", std::fstream::out | std::fstream::app);
    //statFile << gridSize << "," << countThreads << "," << execTime.count() << endl;
    //statFile.close();
    return 0;
}
