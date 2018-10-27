// Lazy-evaluated matrix module in C++
// This file is licensed under the terms of the MIT license.
// 10/26/18, dzchoi, feature-complete
//
// -std=c++17 is required for:
// - class template argument deduction and deduction guides.
// - std::invoke_result_t<>

// Features:
// - Lazy-evaluated matrices can save space; why to allocate memory for Id matrices?
// - Lazy-evaluated and eager-evaluated matrices can be mixed together seamlessly. A 
//   lazy-evaluated matrix holds an expression (called a thunk) that will eventually 
//   compute the resulting matrix when using subscript operator on it like A(i, j) or 
//   assigning it to an eager-evaluated matrix like "B = A".
// - Matrix expression can include mutable matrices, but mutable matrices are captured as 
//   const references, not painfully copied as values.

// Limitation:
// - T from matrix<T> should be an arithmetic type (i.e, an integral or a floating-point 
//   type). As such, this module treats T as a built-in type and does not assume T takes 
//   up some memory, using simple T instead of const T&.
// - Non-const immutable (i.e, non-const lazy-evaluated) matrices cannot be used with 
//   subscript operator(). Declare immutable matrices const explicitly unless there is a 
//   special reason not to.
// - "matrix<T> A = A;" and "matrix<T> A = std::move(A)" will cause runtime error. 
//   However, "A = A;" and "A = std::move(A);" do no harm for a mutable matrix A.

// Todo:
// - Show the user-source line that triggers assert().



#include <cassert>  // assert()
#include <memory>  // unique_ptr<>, make_unique(), get()
#include <type_traits>
    // common_type_t<>, enable_if_t<>, is_arithmetic<>, invoke_result_t<>

#ifdef DEBUG
#include <iostream>  // ostream, cout
#endif



template <typename>
class _thunk_tab;

// _thunk<T> is a function object for evaluating matrix<T> lazily.
template <typename T>
class _thunk {
public:
    unsigned rows, cols;

    virtual T operator()(unsigned, unsigned) const =0;  // evaluator
    virtual ~_thunk() {}
    virtual std::unique_ptr<_thunk> copy() const =0;  // duplicate myself
    virtual _thunk_tab<T>* to_pthunk_tab() { return nullptr; }

#ifdef DEBUG
    virtual std::ostream& show(std::ostream&) const =0;
#endif

protected:
    _thunk(unsigned rows, unsigned cols): rows(rows), cols(cols) {}
};

// turn a _thunk<T> into _thunk_tab<T> if possible.
template <typename T>
inline _thunk_tab<T>& thunk_tab(_thunk<T>& thunk)
{ assert(thunk.to_pthunk_tab()); return *thunk.to_pthunk_tab(); }

#ifdef DEBUG
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const _thunk<T>& thunk)
{ return thunk.show(os); }
#endif



template <typename>
class _thunk_c;

template <typename T>
using _thunk_fn_t = T (*)(unsigned, unsigned);

template <typename>
class _thunk_fn;

template <typename>
class matrix;

template <typename>
class _thunk_map;

template <typename T>
using _thunk_map_t = T (*)(T);

template <typename>
class _thunk_fA;

template <typename T>
using _thunk_fA_t = T (*)(unsigned, unsigned, const matrix<T>&);

template <typename>
class _thunk_fcA;

template <typename T>
using _thunk_fcA_t = T (*)(unsigned, unsigned, T, const matrix<T>&);

template <typename>
class _thunk_fAB;

template <typename T>
using _thunk_fAB_t = T (*)(unsigned, unsigned, const matrix<T>&, const matrix<T>&);

// Note that those function types of _thunk_*_t are defined as a function pointer, which 
// means those functions are "almost pure" and cannot enclose what is determined at 
// runtime, although possibly having a side-effect that is pre-described at compile-time 
// such as outputing to std::cout or generating random numbers.

// sort_of<T> is just T, but disables the template argument deduction for T.
template <typename T>
using sort_of = std::common_type_t<T>;



template <typename T>
class matrix {  // matrix<T> equals just std::unique_ptr<_thunk<T>>.
private:
    std::unique_ptr<_thunk<T>> uthunk;

public:
    const unsigned& rows = uthunk->rows;
    const unsigned& cols = uthunk->cols;

    matrix(const matrix& M)  // creates immutable matrix by copying and referencing.
    : uthunk(M.uthunk->copy()) {}

    matrix(matrix&& M)  // moves M as is.
    : uthunk(std::move(M.uthunk)) {}

    template <typename =std::enable_if_t<std::is_arithmetic<T>{}>>
    matrix(unsigned rows, unsigned cols, T c)
    : uthunk(std::make_unique<_thunk_c<T>>(rows, cols, c)) {}

    matrix(unsigned rows, unsigned cols, _thunk_fn_t<T> fn)
    : uthunk(std::make_unique<_thunk_fn<T>>(rows, cols, fn)) {}

    matrix(sort_of<_thunk_map_t<T>> fn, const matrix& A)
    : uthunk(std::make_unique<_thunk_map<T>>(fn, A)) {}

    matrix(sort_of<_thunk_map_t<T>> fn, matrix&& A)
    : uthunk(std::make_unique<_thunk_map<T>>(fn, std::move(A))) {}

    matrix(unsigned rows, unsigned cols,
	sort_of<_thunk_fA_t<T>> fn, const matrix& A)
    : uthunk(std::make_unique<_thunk_fA<T>>(rows, cols, fn, A)) {}

    matrix(unsigned rows, unsigned cols,
	sort_of<_thunk_fA_t<T>> fn, matrix&& A)
    : uthunk(std::make_unique<_thunk_fA<T>>(rows, cols, fn, std::move(A))) {}

    matrix(unsigned rows, unsigned cols,
	sort_of<_thunk_fcA_t<T>> fn, T c, const matrix<T>& A)
    : uthunk(std::make_unique<_thunk_fcA<T>>(rows, cols, fn, c, A)) {}

    matrix(unsigned rows, unsigned cols,
	sort_of<_thunk_fcA_t<T>> fn, T c, matrix<T>&& A)
    : uthunk(std::make_unique<_thunk_fcA<T>>(rows, cols, fn, c, std::move(A))) {}

    matrix(unsigned rows, unsigned cols,
	sort_of<_thunk_fAB_t<T>> fn, const matrix<T>& A, const matrix<T>& B)
    : uthunk(std::make_unique<_thunk_fAB<T>>(rows, cols, fn, A, B)) {}

    matrix(unsigned rows, unsigned cols,
	sort_of<_thunk_fAB_t<T>> fn, matrix<T>&& A, const matrix<T>& B)
    : uthunk(std::make_unique<_thunk_fAB<T>>(rows, cols, fn, std::move(A), B)) {}

    matrix(unsigned rows, unsigned cols,
	sort_of<_thunk_fAB_t<T>> fn, const matrix<T>& A, matrix<T>&& B)
    : uthunk(std::make_unique<_thunk_fAB<T>>(rows, cols, fn, A, std::move(B))) {}

    matrix(unsigned rows, unsigned cols,
	sort_of<_thunk_fAB_t<T>> fn, matrix<T>&& A, matrix<T>&& B)
    : uthunk(std::make_unique<_thunk_fAB<T>>(
	rows, cols, fn, std::move(A), std::move(B) )) {}

    ~matrix() {}

    T operator()(unsigned i, unsigned j) const {
	assert(i < rows && j < cols); return (*uthunk)(i, j); }

#ifdef DEBUG
    std::ostream& show(std::ostream& os) const;
#endif

public:
    bool is_mutable() const { return bool(uthunk->to_pthunk_tab()); }

    // default constructor, which creates 0x0 mutable matrix.
    matrix()
    : uthunk(std::make_unique<_thunk_tab<T>>()) {}

    T& operator()(unsigned i, unsigned j) {
	// available only for non-const mutable matrices
	assert(i < rows && j < cols); return thunk_tab(*uthunk)(i, j); }

    // The evaluator, which evaluates it and assigns it to a mutable matrix.
    matrix& operator=(const matrix&);
    matrix& operator=(matrix&&);

    // We cannot provide other assignment operators such as +=, because, for example, 
    // when computing "A += A * B", we cannot put the resulting A(0, 0) in place as we 
    // need the old A(0, 0) to compute A(0, 1), A(0, 2), and so on. One way to avoid this 
    // situation is to demand the right-hand side of "+=" to have been evaluated already 
    // like "C = A * B, A += C", but it will not be more efficient than "A = A + A * B". 
    // Similarly, we do not provide mapM() such that A.mapM(f) for some T f(T) changes 
    // each element of A by applying f, because f might be a closure enclosing A in it.
    // It's quite a shame not to be able to use "A += A" instead of "A = A + A". However, 
    // we have no way to make sure the computing of each element does not affect the 
    // computing of other elements.
};

// to make vague "matrix A" into "matrix<double> A".
matrix() -> matrix<double>;

// to be able to deduce T from matrix(unsigned, unsigned, _thunk_fn_t<T>). 
template <typename F>
matrix(unsigned, unsigned, F) -> matrix<std::invoke_result_t<F, unsigned, unsigned>>;

template <typename T>
inline matrix<T>& matrix<T>::operator=(const matrix& M) {
    // We can handle "A = A" though making a redundant copy of A in temporary memory.
    thunk_tab(*uthunk) = *M.uthunk;
    return *this;
}

template <typename T>
inline matrix<T>& matrix<T>::operator=(matrix&& M) {
    if ( M.is_mutable() )
	thunk_tab(*uthunk) = std::move(thunk_tab(*M.uthunk));
    else
	operator=(M);
    return *this;
}

#ifdef DEBUG
template <typename T>
inline std::ostream& matrix<T>::show(std::ostream& os) const {
    return os
	<< "(mutable=" << is_mutable()
	<< ", type=" << typeid(*this->uthunk.get()).name()
	<< ", " << *uthunk
	<< ')';
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const matrix<T>& M)
{ return M.show(os); }

template <typename T>
void show(const matrix<T>& M, const char* header ="")
{
    std::cout << header << M << '\n';
    for ( unsigned i = 0 ; i < M.rows ; ++i ) {
	for ( unsigned j = 0 ; j < M.cols ; ++j )
	    std::cout << (j > 0 ? "\t" : "") << M(i, j);
	std::cout << '\n';
    }
    std::cout << '\n';
}
#endif



template <typename T>
class _thunk_tab_view: public _thunk<T> {
private:
    using _thunk<T>::rows;
    using _thunk<T>::cols;
    const T* table;  // immutable

public:
    _thunk_tab_view(unsigned rows, unsigned cols, const T* table)
    : _thunk<T>(rows, cols), table(table) {}

    T operator()(unsigned i, unsigned j) const override {
	return table[i*cols + j]; }
    std::unique_ptr<_thunk<T>> copy() const override {
	return std::make_unique<_thunk_tab_view>(rows, cols, table); }

#ifdef DEBUG
    std::ostream& show(std::ostream& os) const override {
	return os << "table[]=" << table; }
#endif
};

template <typename T>
class _thunk_tab: public _thunk<T> {
private:
    using _thunk<T>::rows;
    using _thunk<T>::cols;
    std::unique_ptr<T[]> table;

public:
    _thunk_tab(): _thunk<T>(0, 0), table{} {}  // table = nullptr

    T operator()(unsigned i, unsigned j) const override {
	return table[i*cols + j]; }
    T& operator()(unsigned i, unsigned j) {
	return table[i*cols + j]; }
    std::unique_ptr<_thunk<T>> copy() const override {
	// Note that we do not copy _thunk_tab<T> as is, but as a _thunk_tab_view<T>.
	return std::make_unique<_thunk_tab_view<T>>(rows, cols, table.get()); }
    _thunk_tab<T>* to_pthunk_tab() override { return this; }

    void operator=(const _thunk<T>&);
    void operator=(_thunk_tab&&);

#ifdef DEBUG
    std::ostream& show(std::ostream& os) const override {
	return os << "table[]=" << table.get(); }
#endif
};

template <typename T>
void _thunk_tab<T>::operator=(const _thunk<T>& thunk)
{
    std::unique_ptr<T[]> p0 { new T[thunk.rows * thunk.cols] };
    // We do not use make_unique() here because make_unique() initializes each element of 
    // table; see https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique.

    T* p = p0.get();  // to avoid multiplication in accessing each entry in table
    for ( unsigned i = 0 ; i < thunk.rows ; ++i )
	for ( unsigned j = 0 ; j < thunk.cols ; ++j )
	    *p++ = thunk(i, j);

    rows = thunk.rows;
    cols = thunk.cols;
    table = std::move(p0);
}

template <typename T>
inline void _thunk_tab<T>::operator=(_thunk_tab&& thunk) {
    rows = thunk.rows;
    cols = thunk.cols;
    table = std::move(thunk.table);
}

template <typename T>
class _thunk_c: public _thunk<T> {
private:
    using _thunk<T>::rows;
    using _thunk<T>::cols;
    T c;

public:
    _thunk_c(unsigned rows, unsigned cols, T c)
    : _thunk<T>(rows, cols), c(c) {}

    T operator()(unsigned, unsigned) const override { return c; }
    std::unique_ptr<_thunk<T>> copy() const override {
	return std::make_unique<_thunk_c>(rows, cols, c); }

#ifdef DEBUG
    std::ostream& show(std::ostream& os) const override {
	return os << "c=" << c; }
#endif
};

template <typename T>
class _thunk_fn: public _thunk<T> {
private:
    using ftype = _thunk_fn_t<T>;

    using _thunk<T>::rows;
    using _thunk<T>::cols;
    ftype fn;

public:
    _thunk_fn(unsigned rows, unsigned cols, ftype fn)
    : _thunk<T>(rows, cols), fn(fn) {}

    T operator()(unsigned i, unsigned j) const override { return fn(i, j); }
    std::unique_ptr<_thunk<T>> copy() const override {
	return std::make_unique<_thunk_fn>(rows, cols, fn); }

#ifdef DEBUG
    std::ostream& show(std::ostream& os) const override {
	return os << "fn=" << (void*)fn; }
#endif
};

template <typename T>
class _thunk_map: public _thunk<T> {
private:
    using ftype = _thunk_map_t<T>;

    using _thunk<T>::rows;
    using _thunk<T>::cols;
    ftype fn;
    matrix<T> A;

public:
    _thunk_map(ftype fn, const matrix<T>& A)
    : _thunk<T>(A.rows, A.cols), fn(fn), A(A) {}
    _thunk_map(ftype fn, matrix<T>&& A)
    : _thunk<T>(A.rows, A.cols), fn(fn), A(std::move(A)) {}

    T operator()(unsigned i, unsigned j) const override { return fn(A(i, j)); }
    std::unique_ptr<_thunk<T>> copy() const override {
	return std::make_unique<_thunk_map>(fn, A); }

#ifdef DEBUG
    std::ostream& show(std::ostream& os) const override {
	return os << "map=" << (void*)fn << ':' << A; }
#endif
};

template <typename T>
class _thunk_fA: public _thunk<T> {
private:
    using ftype = _thunk_fA_t<T>;

    using _thunk<T>::rows;
    using _thunk<T>::cols;
    ftype fn;
    matrix<T> A;

public:
    _thunk_fA(unsigned rows, unsigned cols, ftype fn, const matrix<T>& A)
    : _thunk<T>(rows, cols), fn(fn), A(A) {}
    _thunk_fA(unsigned rows, unsigned cols, ftype fn, matrix<T>&& A)
    : _thunk<T>(rows, cols), fn(fn), A(std::move(A)) {}

    T operator()(unsigned i, unsigned j) const override { return fn(i, j, A); }
    std::unique_ptr<_thunk<T>> copy() const override {
	return std::make_unique<_thunk_fA>(rows, cols, fn, A); }

#ifdef DEBUG
    std::ostream& show(std::ostream& os) const override {
	return os << "fA=" << (void*)fn << ':' << A; }
#endif
};

template <typename T>
class _thunk_fcA: public _thunk<T> {
private:
    using ftype = _thunk_fcA_t<T>;

    using _thunk<T>::rows;
    using _thunk<T>::cols;
    ftype fn;
    T c;
    matrix<T> A;

public:
    _thunk_fcA(unsigned rows, unsigned cols, ftype fn, T c, const matrix<T>& A)
    : _thunk<T>(rows, cols), fn(fn), c(c), A(A) {}
    _thunk_fcA(unsigned rows, unsigned cols, ftype fn, T c, matrix<T>&& A)
    : _thunk<T>(rows, cols), fn(fn), c(c), A(std::move(A)) {}

    T operator()(unsigned i, unsigned j) const override { return fn(i, j, c, A); }
    std::unique_ptr<_thunk<T>> copy() const override {
	return std::make_unique<_thunk_fcA>(rows, cols, fn, c, A); }

#ifdef DEBUG
    std::ostream& show(std::ostream& os) const override {
	return os << "fcA=" << (void*)fn << ':' << c << ':' << A; }
#endif
};

template <typename T>
class _thunk_fAB: public _thunk<T> {
private:
    using ftype = _thunk_fAB_t<T>;

    using _thunk<T>::rows;
    using _thunk<T>::cols;
    ftype fn;
    matrix<T> A;
    matrix<T> B;

public:
    _thunk_fAB(unsigned rows, unsigned cols,
	ftype fn, const matrix<T>& A, const matrix<T>& B)
    : _thunk<T>(rows, cols), fn(fn), A(A), B(B) {}
    _thunk_fAB(unsigned rows, unsigned cols,
	ftype fn, matrix<T>&& A, const matrix<T>& B)
    : _thunk<T>(rows, cols), fn(fn), A(std::move(A)), B(B) {}
    _thunk_fAB(unsigned rows, unsigned cols,
	ftype fn, const matrix<T>& A, matrix<T>&& B)
    : _thunk<T>(rows, cols), fn(fn), A(A), B(std::move(B)) {}
    _thunk_fAB(unsigned rows, unsigned cols,
	ftype fn, matrix<T>&& A, matrix<T>&& B)
    : _thunk<T>(rows, cols), fn(fn), A(std::move(A)), B(std::move(B)) {}

    T operator()(unsigned i, unsigned j) const override { return fn(i, j, A, B); }
    std::unique_ptr<_thunk<T>> copy() const override {
	return std::make_unique<_thunk_fAB>(rows, cols, fn, A, B); }

#ifdef DEBUG
    std::ostream& show(std::ostream& os) const override {
	return os << "fAB=" << (void*)fn << ':' << A << ':' << B; }
#endif
};



// Id(n): nxn identity matrix
template <typename T>
struct Id: matrix<T> {
    Id(unsigned size) : matrix<T>(size, size, &fn) {}
private:
    static T fn(unsigned i, unsigned j) { return T(i == j); }
};

Id(unsigned) -> Id<double>;



// transpose(A)
template <typename T>
struct transpose: matrix<T> {
    transpose(const matrix<T>& A): matrix<T>(A.cols, A.rows, fn, A) {}
    transpose(matrix<T>&& A): matrix<T>(A.cols, A.rows, fn, std::move(A)) {}
private:
    static T fn(unsigned i, unsigned j, const matrix<T>& A) { return A(j, i); }
};

/* Or, we could define it in a function form:
template <typename T>
inline T _transpose(unsigned i, unsigned j, const matrix<T>& A) {
    return A(j, i);
};

template <typename T>
inline matrix<T> transpose(const matrix<T>& A) {
    return matrix<T>( A.cols, A.rows, _transpose<T>, A );
}

template <typename T>
inline matrix<T> transpose(matrix<T>&& A) {
    return matrix<T>( A.cols, A.rows, _transpose<T>, std::move(A) );
}
*/



// map(fn, A) is a syntactic sugar for matrix(fn, A).
template <typename T, typename F>
inline matrix<T> map(F fn, const matrix<T>& A) {
    // If we defined this as map(_thunk_map_t<T> fn, const matrix<T>& A), fn would take 
    // only functions that exactly match _thunk_map_t<T> without applying any cast.
    return matrix<T>( _thunk_map_t<T>(fn), A );
}

template <typename T, typename F>
inline matrix<T> map(F fn, matrix<T>&& A) {
    return matrix<T>( _thunk_map_t<T>(fn), std::move(A) );
}



// c * A
template <typename T>
inline T _multiply_by_c(unsigned i, unsigned j, T c, const matrix<T>& A) {
    return c * A(i, j);
};

template <typename T>
inline matrix<T> operator*(sort_of<T> c, const matrix<T>& A) {
    // Thanks to sort_of<T> here, we can distinguish "c * A" and "A * B", and we can also 
    // use any types that are convertible to T for c.
    return matrix<T>( A.rows, A.cols, _multiply_by_c<T>, c, A );
}

template <typename T>
inline matrix<T> operator*(sort_of<T> c, matrix<T>&& A) {
    return matrix<T>( A.rows, A.cols, _multiply_by_c<T>, c, std::move(A) );
}



// A + B
template <typename T>
inline T _add(unsigned i, unsigned j, const matrix<T>& A, const matrix<T>& B) {
    return A(i, j) + B(i, j);
}

template <typename T>
inline matrix<T> operator+(const matrix<T>& A, const matrix<T>& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    return matrix<T>( A.rows, A.cols, _add<T>, A, B );
}

template <typename T>
inline matrix<T> operator+(matrix<T>&& A, const matrix<T>& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    return matrix<T>( A.rows, A.cols, _add<T>, std::move(A), B );
}

template <typename T>
inline matrix<T> operator+(const matrix<T>& A, matrix<T>&& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    return matrix<T>( A.rows, A.cols, _add<T>, A, std::move(B) );
}

template <typename T>
inline matrix<T> operator+(matrix<T>&& A, matrix<T>&& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    return matrix<T>( A.rows, A.cols, _add<T>, std::move(A), std::move(B) );
}



// A - B
template <typename T>
inline T _subtract(unsigned i, unsigned j, const matrix<T>& A, const matrix<T>& B) {
    return A(i, j) - B(i, j);
}

template <typename T>
inline matrix<T> operator-(const matrix<T>& A, const matrix<T>& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    return matrix<T>( A.rows, A.cols, _subtract<T>, A, B );
}

template <typename T>
inline matrix<T> operator-(matrix<T>&& A, const matrix<T>& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    return matrix<T>( A.rows, A.cols, _subtract<T>, std::move(A), B );
}

template <typename T>
inline matrix<T> operator-(const matrix<T>& A, matrix<T>&& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    return matrix<T>( A.rows, A.cols, _subtract<T>, A, std::move(B) );
}

template <typename T>
inline matrix<T> operator-(matrix<T>&& A, matrix<T>&& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    return matrix<T>( A.rows, A.cols, _subtract<T>, std::move(A), std::move(B) );
}



// A * B
template <typename T>
T _multiply(unsigned i, unsigned j, const matrix<T>& A, const matrix<T>& B) {
    T sum {};
    for ( unsigned k = 0 ; k < A.cols ; ++k )
	sum += A(i, k) * B(k, j);
    return sum;
}

template <typename T>
inline matrix<T> operator*(const matrix<T>& A, const matrix<T>& B) {
    assert(A.cols == B.rows);
    return matrix<T>( A.rows, B.cols, _multiply<T>, A, B );
}

template <typename T>
inline matrix<T> operator*(matrix<T>&& A, const matrix<T>& B) {
    assert(A.cols == B.rows);
    return matrix<T>( A.rows, B.cols, _multiply<T>, std::move(A), B );
}

template <typename T>
inline matrix<T> operator*(const matrix<T>& A, matrix<T>&& B) {
    assert(A.cols == B.rows);
    return matrix<T>( A.rows, B.cols, _multiply<T>, A, std::move(B) );
}

template <typename T>
inline matrix<T> operator*(matrix<T>&& A, matrix<T>&& B) {
    assert(A.cols == B.rows);
    return matrix<T>( A.rows, B.cols, _multiply<T>, std::move(A), std::move(B) );
}



// schur(A, B): Schur(Hadamard) product
template <typename T>
struct schur: matrix<T> {
    schur(const matrix<T>& A, const matrix<T>& B)
    : matrix<T>( (assert(A.rows == B.rows), A.rows), (assert(A.cols == B.cols), A.cols),
	fn, A, B ) {}
    schur(matrix<T>&& A, const matrix<T>& B)
    : matrix<T>( (assert(A.rows == B.rows), A.rows), (assert(A.cols == B.cols), A.cols),
	fn, std::move(A), B ) {}
    schur(const matrix<T>& A, matrix<T>&& B)
    : matrix<T>( (assert(A.rows == B.rows), A.rows), (assert(A.cols == B.cols), A.cols),
	fn, A, std::move(B) ) {}
    schur(matrix<T>&& A, matrix<T>&& B)
    : matrix<T>( (assert(A.rows == B.rows), A.rows), (assert(A.cols == B.cols), A.cols),
	fn, std::move(A), std::move(B) ) {}

private:
    static T fn(unsigned i, unsigned j, const matrix<T>& A, const matrix<T>& B) {
	return A(i, j) * B(i, j); }
};
