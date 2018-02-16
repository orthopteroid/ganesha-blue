// Copyright 2018 John Howard (orthopteroid@gmail.com)
// Licensed under the MIT License

#ifndef _GANESHA_H_
#define _GANESHA_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/tabulate.h>
#include <thrust/random.h>
#include <thrust/extrema.h>
#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

#include <thrust/system/cuda/detail/cub/device/device_scan.cuh>
#include <thrust/system/cuda/detail/cub/device/device_reduce.cuh>

///////////////

namespace Ganesha {

    enum {
        OptCallback =    0b00000001,
    };

    template<uint16_t IT, uint16_t PS, uint8_t AP, uint8_t AC, uint8_t BP, uint8_t BC, uint8_t CP, uint8_t CC>
    struct BreedingPlan {
        const static uint16_t Iterations = IT;
        const static uint16_t PopSize = PS;
        const static uint8_t AProb = AP;
        const static uint8_t ACount = AC;
        const static uint8_t BProb = BP;
        const static uint8_t BCount = BC;
        const static uint8_t CProb = CP;
        const static uint8_t CCount = CC;
    };

    using DefaultBreedingPlan = BreedingPlan<10, 60000, 5, 1, 10, 1, 50, 2>;

    template<class TData, class TObj, typename BP = DefaultBreedingPlan, uint OPT = 0>
    struct Breeder {
        using BreederType = Breeder<TData, TObj, BP, OPT>;
        const static uint16_t PopSize = BP::PopSize;
        const static uint16_t Iterations = BP::Iterations;
        const static uint Options = OPT;
        const static uint TDataBits = 8 * sizeof(TData);
        const static uint TDataBytes = sizeof(TData);
        const static int PopZoneSize = PopSize / 3;

        using BreedingPlan = BP;

        std::function<uint32_t()> h_SeedFn;
        TData h_best;
        float elapsedMSec = 0.f;
        cudaEvent_t startTimeHandle = 0;
        cudaEvent_t stopTimeHandle = 0;
        uint max_element_position = 0;
        uint iterations = 0;

        thrust::device_vector<TData> d_a, d_b;
        thrust::device_vector<TData> *d_pIn, *d_pOut;
        thrust::device_vector<float> d_fitness, d_fitnessSum;
        thrust::device_vector<uint16_t> d_selections;
        thrust::device_vector<thrust::cuda_cub::cub::KeyValuePair<int, float> > d_kvMax;

        void *d_pSummateTemp = nullptr;
        size_t sizeofSummateTemp = 0;

        void *d_pMaxTemp = nullptr;
        size_t sizeofMaxTemp = 0;

        explicit Breeder(std::function<uint32_t()> hsf) {
            static_assert(BreedingPlan::ACount <= TDataBits, "Ganesha::Solver< ... BreedingPlan::ACount <= TDataBits ... >");
            static_assert(BreedingPlan::BCount <= TDataBits, "Ganesha::Solver< ... BreedingPlan::BCount <= TDataBits ... >");
            static_assert(BreedingPlan::CCount <= TDataBits, "Ganesha::Solver< ... BreedingPlan::CCount <= TDataBits ... >");

            cudaEventCreate(&startTimeHandle);
            cudaEventCreate(&stopTimeHandle);

            h_SeedFn = hsf;

            // prealloc working storage to avoid calling thrust::inclusive_scan, which is wasteful
            thrust::cuda_cub::cub::DeviceScan::DeviceScan::InclusiveSum(
                    d_pSummateTemp, sizeofSummateTemp, d_fitness.data(), d_fitnessSum.data(), PopSize
            );
            cudaMalloc(&d_pSummateTemp, sizeofSummateTemp);

            // prealloc working storage to avoid calling thrust::max_element, which is wasteful
            thrust::cuda_cub::cub::DeviceReduce::ArgMax(
                    d_pMaxTemp, sizeofMaxTemp, d_fitness.data(), d_kvMax.data(), PopSize
            );
            cudaMalloc(&d_pMaxTemp, sizeofMaxTemp);

            d_a.resize(PopSize);
            d_b.resize(PopSize);
            d_fitness.resize(PopSize);
            d_fitnessSum.resize(PopSize);
            d_selections.resize(PopSize);
            d_kvMax.resize(1);

            d_pIn = &d_a;
            d_pOut = &d_b;
        }

        virtual ~Breeder() {
            if (d_pMaxTemp != nullptr) cudaFree(d_pMaxTemp);
            if (d_pSummateTemp != nullptr) cudaFree(d_pSummateTemp);
            cudaEventDestroy(startTimeHandle);
            cudaEventDestroy(stopTimeHandle);
        }

        void TestAndThrow(bool ok)
        {
            if(!ok) throw new std::runtime_error("Ganesha runtime error");
        }

        struct BreedFunctor {
            // nb: no apparent advantage breaking the zones into streams.
            // ie, launch cost of 3 streams greater than overlap.

            uint seed;

            explicit BreedFunctor(uint s) : seed(s) {}

            __device__ TData operator()(const thrust::tuple<TData, int16_t, TData, uint16_t> &arg) {
                TData t;

                // bytewise cross
                // sign of s indicates which item is copied to memory first (ie ordering)
                auto *a = (uint8_t *) &thrust::get<0>(arg);
                int16_t s = thrust::get<1>(arg);
                auto *b = (uint8_t *) &thrust::get<2>(arg);
                if (s < 0) {
                    thrust::swap(a, b);
                    s = -s;
                }
                for (int byte = 0; byte < s; byte++) ((uint8_t *) &t)[byte] = a[byte];
                for (int byte = s; byte < TDataBytes; byte++) ((uint8_t *) &t)[byte] = b[byte];

                // mutate a bit in a byte
                // use different a probability depending upon which third of the population this item is located
                const uint16_t item = thrust::get<3>(arg);

                // population is broken into 3 zones. each with different mutation probabilities.
                const int zoneId = item % PopZoneSize;
                const uint zoneProbability =
                        +(0x01 & (zoneId == 0)) * BreedingPlan::AProb
                        + (0x01 & (zoneId == 1)) * BreedingPlan::BProb
                        + (0x01 & (zoneId == 2)) * BreedingPlan::CProb; // vectorize
                const uint zoneCount =
                        +(0x01 & (zoneId == 0)) * BreedingPlan::ACount
                        + (0x01 & (zoneId == 1)) * BreedingPlan::BCount
                        + (0x01 & (zoneId == 2)) * BreedingPlan::CCount; // vectorize

                thrust::uniform_int_distribution<int> dist;
                thrust::minstd_rand randEng(seed);
                randEng.discard(item); // each item gets its own seed
                for (uint8_t i = 0; i < zoneCount; i++) {
                    randEng.discard(1);
                    if ((((uint) dist(randEng)) % 100) + 1 <= zoneProbability) // +1 to make [1...100]
                    {
                        randEng.discard(1);
                        uint bit = ((uint) dist(randEng)) % TDataBits;
                        ((uint8_t *) &t)[bit / 8] ^= (1 << (bit % 8)); // xor specific bit in specified byte
                    }
                }

                return t;
            }
        };

        ///////////

        struct FloatFunctor {
            thrust::device_ptr<float> d_pUpperNE;
            uint seed;

            FloatFunctor(thrust::device_ptr<float> d_pUNE, uint s) : d_pUpperNE(d_pUNE), seed(s) {}

            // counting_iterator<> and tabulate style
            __device__ float operator()(uint i) {
                thrust::minstd_rand randEng(seed);
                randEng.discard(i); // each item gets its own seed
                thrust::uniform_real_distribution<float> uniDist(0, *d_pUpperNE);
                return uniDist(randEng);
            }
        };

        template<int16_t LEQ, int16_t UEQ>
        struct Int16Functor {
            uint seed;

            explicit Int16Functor(uint s) : seed(s) {}

            // counting_iterator<> and tabulate style
            __device__ int16_t operator()(uint i) {
                thrust::minstd_rand randEng(seed);
                randEng.discard(i); // each item gets its own seed
                thrust::uniform_int_distribution<int> uniDist(LEQ, UEQ + 1);
                return (int16_t)
                uniDist(randEng);
            }
        };

        struct FloatClearFunctor {
            thrust::device_ptr<float> d_pBest;

            explicit FloatClearFunctor(thrust::device_ptr<float> db) : d_pBest(db) {}

            __device__ float operator()(float v) { return v == *d_pBest ? 0.f : v; };
        };

        ///////////////

        void ClearState()
        {
            TestAndThrow( cudaSuccess == cudaMemset(d_a.data().get(), 0, sizeof(TData)*PopSize) );
            TestAndThrow( cudaSuccess == cudaMemset(d_b.data().get(), 0, sizeof(TData)*PopSize) );
            TestAndThrow( cudaSuccess == cudaMemset(d_fitness.data().get(), 0, sizeof(float)*PopSize) );
            TestAndThrow( cudaSuccess == cudaMemset(d_fitnessSum.data().get(), 0, sizeof(float)*PopSize) );
            TestAndThrow( cudaSuccess == cudaMemset(d_selections.data().get(), 0, sizeof(uint16_t)*PopSize) );
        }

        void CopyToDevice(thrust::host_vector<TData>& host_data)
        {
            *d_pIn = host_data; // h2d copy of initial host state
        }

        void CopyToHost(thrust::host_vector<TData>& host_data)
        {
            host_data = *d_pOut; // d2h copy of results to host
        }

        void CopyToHost(TData& h_best, float& h_fitness)
        {
            TestAndThrow( cudaSuccess == cudaMemcpy(&h_best, d_pOut->data().get(), TDataBytes, cudaMemcpyDeviceToHost) );
            TestAndThrow( cudaSuccess == cudaMemcpy(&h_fitness, d_fitness.data().get(), sizeof(float), cudaMemcpyDeviceToHost) );
        }

        void Maximize(std::function<bool(const BreederType *pState)> callbackFn) {
            thrust::counting_iterator<uint16_t> iter0(0), iterN(PopSize); // nb: init issue

            cudaStream_t s1;
            cudaStreamCreate(&s1);

            iterations = 0;

            cudaEventRecord(startTimeHandle, nullptr);

            while(true) {

                /////////////////////////
                // EVAL and PRESERVE BEST

                thrust::transform(
                        //thrust::cuda_cub::execute_on_stream(s1),
                        d_pIn->begin(), d_pIn->end(), d_fitness.begin(), TObj()
                );

                // find max value and key (using a preallocated temporary)
                thrust::cuda_cub::cub::DeviceReduce::ArgMax(
                        d_pMaxTemp, sizeofMaxTemp, d_fitness.data(), d_kvMax.data(), PopSize //, s1
                );

                // avoid having a d2h2d copy by using permutation iterators
                thrust::device_ptr<int> d_pMaxKey = thrust::device_pointer_cast(&(d_kvMax.data().get()->key));
                auto maxKeyIter = thrust::detail::make_normal_iterator(d_pMaxKey);
                auto fitnessMaxIter = thrust::make_permutation_iterator(d_fitness.begin(), maxKeyIter);
                auto inMaxIter = thrust::make_permutation_iterator(d_pIn->begin(), maxKeyIter);

                // copy data and fitness to [0] (d2d version)
                thrust::copy_n(
                        //thrust::cuda_cub::execute_on_stream(s1),
                        thrust::make_zip_iterator(thrust::make_tuple(fitnessMaxIter, inMaxIter)),
                        1,
                        thrust::make_zip_iterator(
                                thrust::make_tuple(d_fitness.begin(), d_pIn->begin()))
                );

                /////////////////////////
                // BOOKEEP

                // consider thrust::swap_ranges(...);
                std::swap(d_pIn, d_pOut); // most recently evaluated are now in pOut
                ++iterations;

                if (Options & OptCallback) {
                    if(callbackFn(this)) break;
                } else {
                    if(iterations >= Iterations) break;
                }

                /////////////////////////
                // CLEAR OTHER-BEST, SUMMATE and SAMPLE

                // clear duplicate 'best' elsewhere
                thrust::transform(
                        //thrust::cuda_cub::execute_on_stream(s1),
                        d_fitness.begin() + 1, d_fitness.end(), d_fitness.begin() + 1, // +1 don't overwrite best
                        FloatClearFunctor(d_fitness.data())
                );

                // parallel prefix sum (using a preallocated temporary)
                thrust::cuda_cub::cub::DeviceScan::InclusiveSum(
                        d_pSummateTemp, sizeofSummateTemp, d_fitness.data(), d_fitnessSum.data(), PopSize //, s1
                );

                // stochastic remainder selection
                FloatFunctor selectionFunctor(d_fitnessSum.data() + PopSize - 1, h_SeedFn());
                thrust::upper_bound(
                        //thrust::cuda_cub::execute_on_stream(s1),
                        d_fitnessSum.begin(), d_fitnessSum.end(),
                        thrust::make_transform_iterator(iter0, selectionFunctor), // iter0 includes best
                        thrust::make_transform_iterator(iterN, selectionFunctor),
                        d_selections.begin()
                );

                ////////////////////////////////
                // CROSSOVER / MUTATE

                // crossover uses a population-index-selection with a (signed) byte-index splice.
                // cross each population member with an expected-selection, as determined in the last step.
                // mutation probabilities apply to thirds of the population.

                // note on index offsets: +1 is used to prevent clobbering the 'best' already pre-written
                // value to the output vector at [0] while -1 is used to reduce the number of items read from the
                // input vectors and iterators. This is all to ensure the number of elements read is the
                // same as the number of elements written.
                using spliceFunctor = Int16Functor<(int16_t) -TDataBytes, (int16_t) +TDataBytes>;
                auto spliceIter = thrust::make_transform_iterator(iter0, spliceFunctor(h_SeedFn()));
                auto spliceBeginIter = thrust::make_permutation_iterator(d_pOut->begin(), d_selections.begin());
                auto spliceEndIter = thrust::make_permutation_iterator(d_pOut->end() - 1, d_selections.end() - 1);
                thrust::transform(
                        //thrust::cuda_cub::execute_on_stream(s1),
                        thrust::make_zip_iterator(
                                thrust::make_tuple(d_pOut->begin(), spliceIter, spliceBeginIter, iter0)),
                        thrust::make_zip_iterator(
                                thrust::make_tuple(d_pOut->end() - 1, spliceIter, spliceEndIter, iterN - 1)),
                        d_pIn->begin() + 1, // produce next input population, +1 don't overwrite [0]
                        BreedFunctor(h_SeedFn())
                );
            }

            cudaEventRecord(stopTimeHandle, nullptr); // stop timer
            cudaEventSynchronize(stopTimeHandle);
            cudaEventElapsedTime(&elapsedMSec, startTimeHandle, stopTimeHandle);

            cudaStreamDestroy(s1);
        }
    };

}

#endif // _GANESHA_H_