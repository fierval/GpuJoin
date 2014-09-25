#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/transform_scan.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/for_each.h>

#include <memory>

#include "utils.h"

#pragma warning (disable : 4367)

struct printf_functor
{
    __host__ __device__
        void operator()(int x)
    {
        // note that using printf in a __device__ function requires
        // code compiled for a GPU with compute capability 2.0 or
        // higher (nvcc --arch=sm_20)
        printf("%d ", x);
    }
};

struct Config
{
    const int minComp;
    const float fractionPatents;
    const int nCompanies; //number of companies in the configuration
    const int * const companyCodes; //array of company codes (nCompanies in number)

    __host__ Config(int minComp, float fractionPatents, int n, const int * const codes) :
        minComp(minComp), fractionPatents(fractionPatents), nCompanies(n), companyCodes(codes) {};

};

typedef thrust::device_vector<int>::iterator   IntIterator;
typedef thrust::device_vector<bool>   BoolVector;

typedef thrust::device_vector<bool>::iterator   BoolIterator;
typedef thrust::tuple<IntIterator, IntIterator> IntIteratorTuple;
typedef thrust::zip_iterator<IntIteratorTuple> ZipIterator;
typedef thrust::device_vector<int>  IntVector;

typedef thrust::host_vector<int> IntVectorHost;
typedef thrust::host_vector<bool> BoolVectorHost;

// iterate over the struct of arrays representing PatentInfo
typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator, IntIterator, BoolIterator, BoolIterator> PatentIteratorTuple;
typedef thrust::zip_iterator<PatentIteratorTuple> PatentZipIterator;
typedef thrust::tuple<int, int, int, int, int, bool, bool> PatentTuple;

struct PatentInfoHost
{
    int size;
    std::vector<int> patentId;
    std::vector<int> classCode;
    std::vector<int> appYear;
    std::vector<int> issueYear;
    std::vector<int> companyId;
    std::vector<bool> isIv;

    PatentInfoHost(int size) : size(size), patentId(size), classCode(size), appYear(size), issueYear(size), companyId(size), isIv(size)
    {
    }
 };

extern "C"
{
    __declspec(dllexport) void GetData(int size, int patetnIds[], int classCodes[], int appYear[], int issueYear[], int companyId[], bool isIv[]);
}

struct PatentInfo
{
    IntVector patentId;
    IntVector classCode;
    IntVector appYear;
    IntVector issueYear;
    IntVector companyId;
    BoolVector isIv;

    PatentInfo(int size, int * patentId, int * classCode, int * appYear, int * issueYear, int * companyId, bool * isIv) : patentId(size), classCode(size), appYear(size),
        issueYear(size), companyId(size), isIv(size)
    {
        copy(patentId, size, this->patentId);
        copy(classCode, size, this->classCode);
        copy(appYear, size, this->appYear);
        copy(issueYear, size, this->issueYear);
        copy(companyId, size, this->companyId);
        copy(isIv, size, this->isIv);

    }

    PatentInfo(int size) : patentId(size), classCode(size), appYear(size),
        issueYear(size), companyId(size), isIv(size) {}

    PatentInfo(PatentInfoHost& patentInfo) : PatentInfo(patentInfo.size) 
    {
        thrust::copy(patentInfo.patentId.begin(), patentInfo.patentId.end(), this->patentId.begin());
        thrust::copy(patentInfo.classCode.begin(), patentInfo.classCode.end(), this->classCode.begin());
        thrust::copy(patentInfo.issueYear.begin(), patentInfo.issueYear.end(), this->issueYear.begin());
        thrust::copy(patentInfo.appYear.begin(), patentInfo.appYear.end(), this->appYear.begin());
        thrust::copy(patentInfo.companyId.begin(), patentInfo.companyId.end(), this->companyId.begin());
        thrust::copy(patentInfo.isIv.begin(), patentInfo.isIv.end(), this->isIv.begin());
    }

    PatentInfo(PatentInfo&& move)
    {
        patentId = move.patentId;
        classCode = move.classCode;
        appYear = move.appYear;
        issueYear = move.issueYear;
        companyId = move.companyId;
        isIv = move.isIv;
    }

    std::unique_ptr<PatentInfoHost> ToHost()
    {
        std::unique_ptr<PatentInfoHost> host(new PatentInfoHost((int)classCode.size()));

        copyToHost(patentId, host->patentId);
        copyToHost(classCode, host->classCode);
        copyToHost(issueYear, host->issueYear);
        copyToHost(appYear, host->appYear);
        copyToHost(companyId, host->companyId);
        copyToHost(isIv, host->isIv);
        return host;
    }

    struct is_matching : public thrust::unary_function < PatentTuple, bool >
    {
        __host__ __device__
            bool operator() (const PatentTuple& patentTuple)
        {
            return thrust::get<6>(patentTuple);
        }
    };
};

class IgapAlgorithm
{
private:
    PatentInfo d_patentInfo;

    int threadCount;

public:

    __host__ IgapAlgorithm(PatentInfoHost& h_patentInfo) :
        d_patentInfo(h_patentInfo), threadCount(1024)
    {
        
    }
    __host__ IgapAlgorithm(PatentInfo&& d_patentInfo) : d_patentInfo(d_patentInfo) {}

    std::unique_ptr<PatentInfoHost> ExpandRelevantClasses(int * const classes, const int size, const Config& config);
};
