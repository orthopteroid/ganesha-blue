// Copyright 2018 John Howard (orthopteroid@gmail.com)
// Licensed under the MIT License

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <memory>
#include <iomanip>
#include <byteswap.h>

// for jetbrains' clion ide
// sssh! requires hack to FindCUDA.cmake file:
// add 'OR ${file} MATCHES "\\.cpp$"' to line starting with 'if((${file} MATCHES "\\.cu$"'
#ifdef __JETBRAINS_IDE__
#define __device__
#define __host__
#define __shared__
#define __global__
#define __device_builtin__
#define tex1D(t,v) (0)
template<typename A, size_t B, size_t C = 0> struct texture {};
inline void __syncthreads() {}
#endif

#include "ganesha.hpp"

////////////

// the state-representation for the objective function
struct Foo
{
    float v;

    __host__ __device__ Foo() : v(0) {}
};

// objective to maximize
// only +ve values allowed as their sum is used to compute a probability density function
struct FooFunc
{
    constexpr static float mu = 101.10101f; // target

    __host__ __device__ float operator()(Foo &foo) const
    {
        float x = foo.v;
        const float delta = 20.f;
        return (1.f/sqrtf(2.f*delta*delta*(float)M_PI))*expf(-(x-mu)*(x-mu)/(2.f*delta*delta));
    };

};

template<int L, int U>
struct FooRandomizer
{
    uint seed;

    explicit FooRandomizer(uint s_) : seed(s_) {}

    // counting_iterator<> and tabulate style
    __host__ __device__ Foo operator()(uint i) {
        thrust::minstd_rand randEng(seed); randEng.discard(i); // each item gets its own seed
        thrust::uniform_real_distribution<float> uniDist( (float)L, (float)U );
        Foo f;
        f.v = uniDist(randEng);
        return f;
    }
};

int main(int argc, char** argv)
{
    // host seed-function
    timespec ts = {0,0};
    clock_gettime( CLOCK_MONOTONIC, &ts );
    std::default_random_engine generator((uint32_t)ts.tv_sec ^ bswap_32((uint32_t)ts.tv_nsec));
    std::uniform_int_distribution<uint32_t> distribution(0);
    std::function<uint32_t()> seedFn = [&] () { return distribution(generator); };

    // optimizer configuration template
    using FooBreeder = Ganesha::Breeder<
        Foo,        // storage or state structure for objective function
        FooFunc     // objective function used for maximization. must return only +ve values.
        ,Ganesha::DefaultBreedingPlan, Ganesha::OptCallback
    >;

    std::cout << std::fixed << std::setprecision(5); // set format for all floats

    {
        FooBreeder solver(seedFn);
        thrust::host_vector<Foo> h_data(solver.PopSize);

        for (int i = 0; i < 10; i++) {
            Foo best;
            float lastfit = std::numeric_limits<float>::min();
            float newfit = 0;
            float percent = 0;

            solver.ClearState();

            thrust::tabulate(h_data.begin(), h_data.end(), FooRandomizer<-10000, +10000>(seedFn()));
            solver.CopyToDevice(h_data);

            solver.Maximize(
                    // callbackFn function. called after each iteration. return true to stop iterating
                    [&](const FooBreeder *pState) -> bool {
                        lastfit = newfit;
                        solver.CopyToHost(best, newfit);
                        char hackChar = (newfit < lastfit ? '-' : ' '); // indicate monotonic issues
                        percent = 100.f * std::abs(best.v - FooFunc::mu) / FooFunc::mu;
                        std::cout << hackChar << percent << ", ";
                        return percent < .01f;
                    }
            );

            solver.CopyToHost(best, newfit);
            std::cout << best.v << ", " << solver.elapsedMSec / solver.iterations << ", " << solver.iterations << '\n';
        }
    }
    return 0;
}
