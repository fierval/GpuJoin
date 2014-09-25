#pragma warning (disable : 4267)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "algorithm.h"
#include <algorithm>
#include "timer.h"
#include <ctime>

int main()
{

    int repeat = 1000000;
    int coreSize = 20;
    int size = coreSize * repeat;

    PatentInfoHost hostInfo(size);

    std::generate(hostInfo.patentId.begin(), hostInfo.patentId.end(), std::rand);
    std::generate(hostInfo.companyId.begin(), hostInfo.companyId.end(), std::rand);
    std::generate(hostInfo.issueYear.begin(), hostInfo.issueYear.end(), std::rand);
    std::generate(hostInfo.appYear.begin(), hostInfo.appYear.end(), std::rand);
    std::fill(hostInfo.isIv.begin(), hostInfo.isIv.end(), false);

    std::generate(hostInfo.classCode.begin(), hostInfo.classCode.end(), std::rand);

    int classSize = 4000;
    std::vector<int> classes(classSize);
    std::generate(classes.begin(), classes.end(), std::rand);


    Config config(2, 0.9f, 0, NULL);

    IgapAlgorithm igap(hostInfo);

    timer t;

    std::unique_ptr<PatentInfoHost> res = igap.ExpandRelevantClasses(classes.data(), sizeof(classes) / sizeof(int), config);

    printf("cuda elapsed: %f\n", t.elapsed());
    //std::for_each(res->classCode.begin(), res->classCode.end(), printf_functor());
    printf("\n");

    clock_t start = clock();

    std::vector<bool> manual(size);
    std::fill(manual.begin(), manual.end(), false);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < classes.size(); j++)
        {
            bool match = classes[j] == hostInfo.classCode[i];
            if (match)
            {
                manual[i] = match;
            }
        }
    }
    clock_t end = clock();

    printf("cpu elapsed: %f\n", static_cast<double>(end - start) / static_cast<double>(CLOCKS_PER_SEC));

    std::cout << '\n';

    return 0;
}
