#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <immintrin.h>
#include <functional>

constexpr int size_gb = 12;
constexpr uint64_t size = 1024 * 1024 * 1024* (uint64_t)size_gb;
constexpr int workers_number{8};
constexpr int calls{1048576}; // 2**20
constexpr int repetitions{8};

struct timed_scope {
    std::string name;
    std::chrono::high_resolution_clock::time_point start;

    timed_scope(const std::string &name) : start( std::chrono::high_resolution_clock::now() ), name(name) {}

    ~timed_scope() {
        auto took = std::chrono::duration_cast<std::chrono::microseconds>
                (std::chrono::high_resolution_clock::now() - start).count();
        std::cout << name << " -- took: " <<
                  took/1000.0 << " ms [" << size / (took * 1e3) << "GB/s]" << std::endl;
    }

};



void init_buffers(void *&buf1, void *&buf2) {

    buf1 = malloc(size);

    if (!buf1) {
        std::cout << "Error in malloc" << std::endl;
        exit(1);
    }

    buf2 = malloc(size);

    if (!buf2) {
        std::cout << "Error in malloc" << std::endl;
        exit(1);
    }


}

void free_buffers(void *buf1, void *buf2) {

    free(buf1);
    buf1 = nullptr;

    free(buf2);
    buf2 = nullptr;

}

void single_thread_memcpy() {

    void *dst, *src;
    init_buffers(dst, src);

    {
        timed_scope _("Single thread memcpy ");
        memcpy(dst, src, size);
    }

    auto res = memcmp(dst, src, size);

    if (res) {
        std::cout << "Error in compare" << std::endl;
        exit(1);
    }

    std::cout << "Compare ok" << std::endl;

    free_buffers(dst, src);
}

void single_thread_memcpy_multiple() {

    void *dst, *src;
    init_buffers(dst, src);

    size_t per_call_size{size / calls};

    {
        timed_scope _("Single thread memcpy multicall");
        for (auto i = 0; i < calls; ++i) {
            memcpy(
                ((uint8_t *) dst) + i * per_call_size,
                ((uint8_t *) src) + i * per_call_size,
                per_call_size
            );
        }
    }

    auto res = memcmp(dst, src, size);

    if (res) {
        std::cout << "Error in compare" << std::endl;
        exit(1);
    }

    std::cout << "Compare ok" << std::endl;

    free_buffers(dst, src);
}


template<typename Functor>
void multithread_copy_helper(Functor function, const std::string &name) {

    void *dst, *src;
    init_buffers(dst, src);

    size_t per_worker_size{size / workers_number};

    std::vector<std::thread> workers;
    workers.reserve(workers_number);

    {
        timed_scope _("Multithread " + name + " thread memcpy");

        for (auto i = 0; i < workers_number; ++i) {
            workers.emplace_back(
                    [i, function, dst, src, per_worker_size]() {
                        function(
                                ((uint8_t *) dst) + i * per_worker_size,
                                ((uint8_t *) src) + i * per_worker_size,
                                per_worker_size
                        );
                    }
            );
        }

        for (auto &thread : workers)
            if (thread.joinable())
                thread.join();

    }

    auto res = std::memcmp(dst, src, size);
    if (res) {
        std::cout << "Error in compare" << std::endl;
        exit(1);
    }
    std::cout << "Compare ok" << std::endl;

    free_buffers(dst, src);
}

// dst and src must be 16-byte aligned
// size must be multiple of 16*8 = 128 bytes
static void copy_with_sse(uint8_t *dst, uint8_t *src, size_t size) {
    size_t stride = 8 * sizeof(__m128i);
    while (size) {
        __m128 a = _mm_load_ps((float *) (src + 0 * sizeof(__m128)));
        __m128 b = _mm_load_ps((float *) (src + 1 * sizeof(__m128)));
        __m128 c = _mm_load_ps((float *) (src + 2 * sizeof(__m128)));
        __m128 d = _mm_load_ps((float *) (src + 3 * sizeof(__m128)));
        __m128 e = _mm_load_ps((float *) (src + 4 * sizeof(__m128)));
        __m128 f = _mm_load_ps((float *) (src + 5 * sizeof(__m128)));
        __m128 g = _mm_load_ps((float *) (src + 6 * sizeof(__m128)));
        __m128 h = _mm_load_ps((float *) (src + 7 * sizeof(__m128)));
        _mm_store_ps((float *) (dst + 0 * sizeof(__m128)), a);
        _mm_store_ps((float *) (dst + 1 * sizeof(__m128)), b);
        _mm_store_ps((float *) (dst + 2 * sizeof(__m128)), c);
        _mm_store_ps((float *) (dst + 3 * sizeof(__m128)), d);
        _mm_store_ps((float *) (dst + 4 * sizeof(__m128)), e);
        _mm_store_ps((float *) (dst + 5 * sizeof(__m128)), f);
        _mm_store_ps((float *) (dst + 6 * sizeof(__m128)), g);
        _mm_store_ps((float *) (dst + 7 * sizeof(__m128)), h);

        size -= stride;
        src += stride;
        dst += stride;
    }
}

void single_thread_sse() {

    void *dst, *src;
    init_buffers(dst, src);

    {
        timed_scope _("copy_with_sse memcpy ");

        copy_with_sse( (uint8_t *) dst,
                       (uint8_t *) src,
                       size            );
    }

    auto res = memcmp(dst, src, size);

    if (res) {
        std::cout << "Error in compare" << std::endl;
        exit(1);
    }

    std::cout << "Compare ok" << std::endl;

    free_buffers(dst, src);
}


int main() {
    std::cout << "Memcpy test -- copying " << size_gb << "GB " << std::endl;

    for (auto i = 0; i < repetitions; i++)
    {
        single_thread_sse();
    }

    for (auto i = 0; i < repetitions; i++)
    {
        single_thread_memcpy();
    }

    for (auto i = 0; i < repetitions; i++)
    {
        single_thread_memcpy_multiple();
    }
    
    for (auto i = 0; i < repetitions; i++)
    {
        multithread_copy_helper(memcpy, "memcpy");
    }

    for (auto i = 0; i < repetitions; i++)
    {
        multithread_copy_helper(copy_with_sse, "sse");
    }    
    
}
