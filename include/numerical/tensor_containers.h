#pragma once

#include "common.h"

#include <vector>
#include <cstdlib>
#include <cassert>

namespace numerical {

    template<typename T>
    struct PODVector {
        T* data;
        int n;

        PURE int size() const {
            return n;
        }

        PURE T& operator()(int idx) {
            assert (idx >= 0 && idx < n);
            return data[idx];
        }

        PURE const T& operator()(int idx) const {
            return const_cast<PODVector*>(this)->operator()(idx);
        }

        PURE T& operator[](int idx) {
            return operator()(idx);
        }

        PURE const T& operator[](int idx) const {
            return operator()(idx);
        }
    };

    template<typename T, bool RowMajor>
    struct PODMatrix {
        T* data;
        int n, m;

        PURE T& operator()(int row, int col) {
            assert (row >= 0 && row < n && col >= 0 && col < m);
            return data[RowMajor ? row * m + col : col * n + row];
        }

        PURE const T& operator()(int row, int col) const {
            return const_cast<PODMatrix*>(this)->operator()(row, col);
        }

        PURE T& operator[](int idx) {
            return data[idx];
        }

        PURE const T& operator[](int idx) const {
            return data[idx];
        }

        PURE int rows() const {
            return n;
        }

        PURE int cols() const {
            return m;
        }

        PURE int size() const {
            return n * m;
        }
    };

    template<class T>
    struct CudaAllocator
    {
        typedef T value_type;
    
        CudaAllocator() = default;
    
        template<class U>
        constexpr CudaAllocator(const CudaAllocator <U>&) noexcept {}
    
        [[nodiscard]] T* allocate(std::size_t n)
        {
            T* p;
            CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
            report(p, n);
    
            throw std::bad_alloc();
        }
    
        void deallocate(T* p, std::size_t n) noexcept
        {
            CUDA_CHECK(cudaFree(this->data));
            report(p, n, 0);
        }
    
    private:
        void report(T* p, std::size_t n, bool alloc = true) const
        {
            // std::cout << (alloc ? "Alloc: " : "Dealloc: ") << sizeof(T) * n
            //         << " bytes at " << std::hex << std::showbase
            //         << reinterpret_cast<void*>(p) << std::dec << '\n';
        }
    };

    template<class T, class U>
    bool operator==(const CudaAllocator <T>&, const CudaAllocator <U>&) { return true; }
    
    template<class T, class U>
    bool operator!=(const CudaAllocator <T>&, const CudaAllocator <U>&) { return false; }


    template<class T>
    struct Mallocator
    {
        typedef T value_type;
    
        Mallocator() = default;
    
        template<class U>
        constexpr Mallocator(const Mallocator <U>&) noexcept {}
    
        [[nodiscard]] T* allocate(std::size_t n)
        {
            if (auto p = static_cast<T*>(std::malloc(n * sizeof(T))))
            {
                report(p, n);
                return p;
            }
    
            throw std::bad_alloc();
        }
    
        void deallocate(T* p, std::size_t n) noexcept
        {
            report(p, n, 0);
            std::free(p);
        }
    
    private:
        void report(T* p, std::size_t n, bool alloc = true) const
        {
            // std::cout << (alloc ? "Alloc: " : "Dealloc: ") << sizeof(T) * n
            //         << " bytes at " << std::hex << std::showbase
            //         << reinterpret_cast<void*>(p) << std::dec << '\n';
        }
    };

    template<class T, class U>
    bool operator==(const Mallocator <T>&, const Mallocator <U>&) { return true; }
    
    template<class T, class U>
    bool operator!=(const Mallocator <T>&, const Mallocator <U>&) { return false; }

    template<typename T, class Allocator = Mallocator<T>>
    struct ManagedVector : PODVector<T> {
        Allocator allocator;

        ManagedVector(int n, const Allocator& allocator = Allocator()) : allocator(allocator) {
            this->n = n;
            this->data = this->allocator.allocate(this->size());
        }

        ~ManagedVector() {
            allocator.deallocate(this->data, this->size());
        }
    };

    template<typename T, bool RowMajor, class Allocator = Mallocator<T>>
    struct ManagedMatrix : PODMatrix<T, RowMajor> {
        Allocator allocator;

        ManagedMatrix(int n, int m, const Allocator& allocator = Allocator()) : allocator(allocator) {
            this->n = n;
            this->m = m;
            this->data = this->allocator.allocate(this->size());
        }

        ~ManagedMatrix() {
            allocator.deallocate(this->data, this->size());
        }
    };
}
