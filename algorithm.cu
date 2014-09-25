#include "algorithm.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <iostream>
#include "timer.h"

std::unique_ptr<IgapAlgorithm> igapAlgorithm;

__global__ void joinValue(const int left, const int * const right, int rightSize, bool * const matches)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rightSize)
    {
        bool match = left == right[idx];
        if (match)
        {
            matches[idx] = match;
        }
    }
}

__global__ void join(const int * const left, const int * const right, int leftSize, int rightSize, int childBlocks, int childThreadsPerBlock, bool * const matches)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < leftSize)
    {
        joinValue<<<childBlocks, childThreadsPerBlock>>>(left[idx], right, rightSize, matches);
    }

}

std::unique_ptr<PatentInfoHost> IgapAlgorithm::ExpandRelevantClasses(int * const classes, const int size, const Config& config)
{
    timer t;
    IntVector upcs(size);
    copy(classes, size, upcs);

    // allocate an array which will signify succesful joins
    BoolVector matches(d_patentInfo.classCode.size());
    thrust::fill(matches.begin(), matches.end(), false);

    // extract a pointer to class codes and invoke a join.
    const int * const classCodesRaw = thrust::raw_pointer_cast(d_patentInfo.classCode.data());
    bool * const matchesRaw = thrust::raw_pointer_cast(matches.data());
    const int * const upcsRaw = thrust::raw_pointer_cast(upcs.data());

    int nBlocks = (size + threadCount) / threadCount;
    int nChildBlocks = ((int)d_patentInfo.classCode.size() + threadCount) / threadCount;

    // launch join kernel
    int rightSize = (int)d_patentInfo.classCode.size();
    join<<<nBlocks, threadCount>>>(upcsRaw, classCodesRaw, size, rightSize, nChildBlocks, threadCount, matchesRaw);

    //... and wait for it to finish
    cudaDeviceSynchronize();
    printf("join elapsed event: %f\n", t.elapsed());

    //thrust::for_each(matches.begin(), matches.end(), printf_functor());
    //std::cout << '\n';

    int nMatches = (int)thrust::count(matches.begin(), matches.end(), true);
    printf("matches: %d\n", nMatches);

    PatentInfo relevantPatents(nMatches);

    // output iterator. "matches" at the end has a larger size than nMatches.
    PatentZipIterator relevantPatentsIter = thrust::make_zip_iterator(thrust::make_tuple(relevantPatents.patentId.begin(), relevantPatents.classCode.begin(), relevantPatents.appYear.begin(), relevantPatents.issueYear.begin(), relevantPatents.companyId.begin(), relevantPatents.isIv.begin(), matches.begin()));

    //input iterator
    PatentZipIterator patentZipIter = thrust::make_zip_iterator(thrust::make_tuple(d_patentInfo.patentId.begin(), d_patentInfo.classCode.begin(), d_patentInfo.appYear.begin(), d_patentInfo.issueYear.begin(), d_patentInfo.companyId.begin(), d_patentInfo.isIv.begin(), matches.begin())); 

    PatentZipIterator patentZipIterEnd = thrust::make_zip_iterator(thrust::make_tuple(d_patentInfo.patentId.end(), d_patentInfo.classCode.end(), d_patentInfo.appYear.end(), d_patentInfo.issueYear.end(), d_patentInfo.companyId.end(), d_patentInfo.isIv.end(), matches.end()));

    //copy to the output if we are matching
    thrust::copy_if(patentZipIter, patentZipIterEnd, relevantPatentsIter, PatentInfo::is_matching());

    //copy to the host and return
    return relevantPatents.ToHost();
}

__declspec(dllexport) void GetData(int size, int patentIds[], int classCodes[], int appYear[], int issueYear[], int companyId[], bool isIv[])
{
    igapAlgorithm = std::unique_ptr<IgapAlgorithm>(new IgapAlgorithm(PatentInfo(size, patentIds, classCodes, appYear, issueYear, companyId, isIv)));
}