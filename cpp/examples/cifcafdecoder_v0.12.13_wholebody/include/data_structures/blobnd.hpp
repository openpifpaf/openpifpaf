#pragma once

#include <functional>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace std;


template<typename T>
class blobnd {
private:
    std::vector<unsigned> strides_;
    T* data_ = NULL;
public:
    std::vector<unsigned> shape;
    unsigned size = 0;
    unsigned dim = 0;

private:
    unsigned         _get_flat_index(const vector<unsigned> &indices) const;
public:
    /*************************************************************************
    *                             Object Lifecycle                           *
    *************************************************************************/
    blobnd();
    blobnd(const vector<unsigned> &shape);
    blobnd(const T &v, const vector<unsigned> &shape);
    blobnd(T *data, const vector<unsigned> &shape);
    blobnd(const blobnd<T> &copy);
    ~blobnd();
    blobnd<T>& operator=(const blobnd<T> &a);
    /*************************************************************************/

    /*************************************************************************
    *                             Get / Set                                  *
    *************************************************************************/
    T*               data() const;
    vector<unsigned> strides() const;
    void             _set_strides();
    /************************************************************************/

    /*************************************************************************
    *                    (i, j, k, ...) get and set                          *
    *************************************************************************/
    template<typename... I>
    T           operator()(I... indices) const;
    template<typename... I>
    T&          operator()(I... indices);
    T           at(const vector<unsigned> &indices) const;
    T&          at(const vector<unsigned> &indices);
    /*************************************************************************/
};


/***********************************************************************************************************************
 *                                               Object Lifecycle                                                      *
 **********************************************************************************************************************/
template <typename T>
blobnd<T>::blobnd() {

}

template <typename T>
blobnd<T>::~blobnd() {
    delete[] data_;
    data_ = NULL;
    dim = size = 0;
}

template <typename T>
blobnd<T>::blobnd(const blobnd<T> &copy) {
    dim = copy.dim;
    shape = copy.shape;
    strides_ = copy.strides_;
    size = copy.size;

    data_ = new T[copy.size];
    std::copy(copy.data_, copy.data_ + copy.size, data_);
}

template <typename T>
blobnd<T>& blobnd<T>::operator=(const blobnd<T> &a) {
    if(this != &a) {
        delete[] data_;

        dim = a.dim;
        shape = a.shape;
        strides_ = a.strides_;
        size = a.size;

        data_ = new T[a.size];
        std::copy(a.data_, a.data_ + a.size, data_);
    }

    return *this;
}

template <typename T>
blobnd<T>::blobnd(const vector<unsigned> &shape)
        : shape(shape)
{
    const unsigned shape_product = accumulate(shape.begin(), shape.end(), 1, multiplies<unsigned>());
    size  = shape_product;
    dim   = shape.size();
    data_ = new T[shape_product];
    std::fill(data_, data_ + size, 0);

    _set_strides();
}

template <typename T>
blobnd<T>::blobnd(const T &v, const vector<unsigned> &shape)
        : shape(shape)
{
    const unsigned shape_product = accumulate(shape.begin(), shape.end(), 1, multiplies<unsigned>());
    size  = shape_product;
    dim   = shape.size();
    data_ = new T[shape_product];
    std::fill(data_, data_ + size, v);

    _set_strides();
}

template <typename T>
blobnd<T>::blobnd(T *data, const vector<unsigned> &shape)
        : shape(shape)
{
    const unsigned shape_product = accumulate(shape.begin(), shape.end(), 1, multiplies<unsigned>());
    size = shape_product;
    dim = shape.size();
    data_ = new T[size];
    std::copy(data, data + size,  data_);

    _set_strides();
}
/**********************************************************************************************************************/


/***********************************************************************************************************************
 *                                               Get / Set                                                             *
 **********************************************************************************************************************/
template <typename T>
T* blobnd<T>::data() const{
    return data_;
}

template <typename T>
vector<unsigned> blobnd<T>::strides() const {
    return strides_;
}
/**********************************************************************************************************************/


/***********************************************************************************************************************
 *                                           Internal utility methods                                                  *
 **********************************************************************************************************************/
template <typename T>
void blobnd<T>::_set_strides() {
    if(dim > 1) {
        vector<unsigned> subps;
        for(int i = 1; i < this->dim; i++) {
            unsigned sp = accumulate(shape.begin() + i, shape.end(), 1, multiplies<unsigned>());
            subps.push_back(sp);
        }
        strides_ = subps;
    }
    else
        strides_ = {1};
}

template <typename T>
unsigned blobnd<T>::_get_flat_index(const vector<unsigned> &indices) const {
    unsigned indices_size = indices.size();
    unsigned index = indices[0];

    if(indices_size == 1)
        return index;

    index = (this->shape[1] * indices[0]) + indices[1];

    if(indices_size == 2)
        return index;

    for(int i = 2; i < indices_size; i++) {
        index = index * this->shape[i] + indices[i];
    }

    return index;
}
/**********************************************************************************************************************/


/***********************************************************************************************************************
 *                                         (i, j, k, ...) get and set                                                  *
 **********************************************************************************************************************/
template <typename T>
template<typename... I>
T blobnd<T>::operator()(I... indices) const {
    vector<unsigned> is = {(unsigned)indices...};
    unsigned index = _get_flat_index(is);

    return data_[index];
}

template <typename T>
template<typename... I>
T& blobnd<T>::operator()(I... indices) {
    vector<unsigned> is = {(unsigned)indices...};
    unsigned index = _get_flat_index(is);

    return data_[index];
}

template <typename T>
T blobnd<T>::at(const vector<unsigned> &indices) const {
    unsigned index = _get_flat_index(indices);

    return data_[index];
}

template <typename T>
T& blobnd<T>::at(const vector<unsigned> &indices) {
    unsigned index = this->_get_flat_index(indices);

    return data_[index];
}
/**********************************************************************************************************************/