# Lazy-evaluated matrix module in C++

This lazy-evaluated matrix module is intended to be used for calculations using matrices and writing functions that include matrix computations.

This module is written in C++17, mostly in C++11, to be user-friendly in terms of its usage, and at the same time to be highly efficient in memory handling. You need to have basic knowledge of C++11 such as lambda function, copy-construct, move-construct, copy-assignment, move-assignment, lvalue references and rvalue references, in order to use this module effectively, and in particular to write your own functions.

### C++17 is required

This module requires C++17 for depending on:
* class template argument deduction and deduction guides, and
* std::invoke_result_t<>.

### Lazy evaluation

Most matrix modules evaluate matrices eagerly and in place. They compute matrix expressions and store the results in memory, usually in two-dimensional arrays. However, this computing model may waste memory sometime, when matrices are very simple and easily computable.

For example, 1000x1000 identity matrix will occupy an array of 1,000,000 entries, most of which are nothing but 0.

For another example, when we need `transpose(A)` for a given matrix `A`, we don't actually need a separate matrix in memory that has the same elements of `A` but in slightly different order. We could simply use (j,i)-th element of `A` when we need the (i,j)-the element of `transpose(A)`.

This module can define the 3x3 identity matrix as:
~~~C++
matrix I = matrix(3, 3, [](unsigned i, unsigned j){ return double(i == j); });
~~~

The above matrix `I` holds the lambda function instead of the whole 3x3 two-dimentional array, and computs each of its element on demand:
~~~C++
std::cout << I(0, 0);
~~~

For a zero-matrix that consists of all 0s, we can define it as:
~~~C++
matrix Zero = matrix(3, 3, [](unsigned, unsigned){ return 0.; });  // returning "0." to make it matrix<double>.
~~~
However, we can use a syntactic sugar for that one:
~~~C++
matrix Zero = matrix(3, 3, 0.);
~~~

### Lazy evaluation and eager evaluation can be mixed.

There are two types of matrices in this module, lazy-evaluated matrices and eager-evaluated matrices.

Lazy-evaluated matrix can be simply thought of a matrix expression itself, which is compiled and saved in memory, but not evaluated on the spot. We can bind a lazy-evaluated matrix to a variable using C++ declaration statement like:
~~~C++
matrix Zero = matrix(2, 3, 0.);  // 2x3 matrix of all 0s
matrix One(2, 3, 1.);  // the same as: matrix One = matrix(2, 3, 1.);
matrix Two = 2 * One + Zero;
~~~

Variables of lazy-evaluated matrices are immutable, and as such they are recommended to be declared `const` like:
~~~C++
const matrix Two = 2 * One + Zero;
~~~

They must be initialized, and once initialized they cannot be changed.
~~~C++
matrix Zero = matrix(2, 3, 0.);
matrix One = matrix(2, 3, 1.);
Zero = One;  // runtime assertion error!
~~~

However, eager-evaluated matrix has an associated in-memory array for storing its elements (that is, in C++ terms eager-evaluated matrix always exists as an lvalue). It comes into existence on declaring a variable without initialization, and can be changed using assignment operator `=`:
~~~C++
matrix X;  // mutable (eager-evaluated) matrix
X = matrix(2, 3, 1.);
X = 2 * X;
~~~

As we can see above, we can assign a lazy-evaluated matrix to a eager-evaluated matrix and get the expression held in the lazy-evaluated matrix evaluated. This is the only way to evaluate lazy-evaluated matrices eventually. (Actually, there is another way of evalution, applying a subscript operator to them.)

---
From now on, we call:
* ***immutable matrix***: lazy-evaluated matrix that is bound to a variable or not.
* ***mutable matrix***: eager-evaluated matrix that is (always) bound to a variable.

Note also that:
* immutable matrix variables are created when declaring ***with initialization***.
* mutable matrix variables are created ***without initialization***.
---

### Immutable matrices can depend on mutable matrices.

When we define an immutable matrix from existing immutable matrices (immutables in short), all expressions (called thunks as they are compiled expressions) from existing immutables are copied to build a thunk for the new immutable matrix.<sup>[1]</sup>

Immutable matrices can also depend on existing mutables, and in this case, references of the mutables (that is, `const matrix<T>&`) are used to build the thunk, instead of copying the whole in-memory arrays associated with the mutables. This is actually where the memory-efficiency of this modules comes into play.

However, programmers should take care of these dependencies and should be careful not to break dependencies.

~~~C++
const matrix A = matrix(2, 3, 1.);  // a 2x3 matrix of all 1s
matrix B;
B = A;  // B is 2x3 mutable matrix having a separate array in memory.
const matrix C = B;  // C is an immutable matrix but dependent on B.

std::cout << C(0, 0) << '\n';  // will show 1.
B(0, 0) = -1;  // We can change B as is mutable.
std::cout << C(0, 0) << '\n';  // will show -1.
~~~

<sub>[1] All thunks and sub-thunks from dependent immutables are traced and copied entirely. That means, if we have `matrix C = A + B; matrix D = C;`, it is the same as `matrix C = A + B; matrix D = A + B;` in regarding to the internals of `C` and `D`, and `D` does not make use of `C` for saving some memory space. This is technically because C++ does not have global garbage collector as Python or some other dynamically-typed languages do, and we cannot control the lifetime of dependent matrix variables; we cannot extend the their lifetime just because we are referring to them. However, thunks do not occupy so much amount of memory as arrays in mutable matrices, and we can decide not to make additional matrix variables if we concern the memory space the redundant thunks may occupy.</sub>

### `matrix<T>` is a template.

`matrix<T>` is actually a template, not a single type. It takes its element type as the template argument `T`, which should be an arithmetic type like `int`, `float` and `double`. Mostly, however, we can omit the type `T` and have it deduced from the initializer, thanks to template argument deduction of C++17.
~~~C++
matrix A(3, 3, 1.);  // matrix<double> because of 1. is double
matrix B = matrix(3, 3, 1.);  // same here
matrix C(3, 3, [](unsigned i, unsigned j){return double(i == j); });  // matrix<double> since the lambda returns double.
matrix D = A + C;  // matrix<double> since A and C are matrix<double>.
~~~

Even when we create a mutable matrix, which does not include any information about its elements, we can still omit `T`. In this case, `double` is defaulted.
~~~C++
matrix A;  // matrix<double> by default
matrix<int> B;  // matrix<int> as specified
~~~

### Matrix operations

This module does not provide complete set of matrix operations. You can use the `class matrix<T>` to define your own matrix operations. Basic operations that are provided by this module is:

class members:
* `.rows`, `.cols` : `A.rows` and `A.cols` tell the size of `A`.
* `.operator()` : `A(i, j)` returns the (i,j)-th element of `A`.
* `.is_mutable()` : `A.is_mutable()` returns a `bool` indicating whether `A` is mutable.
* `operator=` : `A = B` and `A = std::move(B)` can copy- and move-assign `B` to `A`, respectively.

non-members:
* `Id(unsigned n)` : nxn identity matrix of `double`
* `Id<T>(unsigned n)` : nxn identity matrix of `T`
* `transpose(matrix<T> A)` : returns a new immutable matrix with `A`transposed
* `matrix(T (*fn)(T), matrix<T> A)` : a new immutable matrix with applying `fn` to each element of `A`
* `map(T (*fn)(T), matrix<T> A)` : syntactic sugar for `matrix(T (*fn)(T), matrix<T> A)`
* `c * A` (where `T c` and `matrix<T> A`) : scalar multiplication of `c` and `A`
* `A + B` : a new immutable matrix by adding two matrices, `A` and `B`
* `A - B` : a new immutable matrix by subtracting `B` from `A`
* `A * B` : a new immutable matrix by multiplying two matrices, `A` and `B`
* `schur(A, B)` : a new immutable matrix of [Schur(Hadamard) product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) between `A` and `B`

### Subscript operator, `operator()(unsigned, unsigned)`

Two kinds of `opeator()` is provided:
* `T operator()(unsigned, unsigned) const`
* `T& operator()(unsigned, unsigned)`

The first one is called on const matrix, while the second is called on non-const matrix, regardless of the matrix is mutable or immutable. However, the second can be called only for mutable matrix and causes an assert error otherwise.
~~~C++
const matrix A = matrix(2, 3, 0.);
std::cout << A(0, 0);  // OK
A(0, 0) = 1.;  // compile-time error; A(0, 0) returns T, which is not assignable.

matrix B = matrix(2, 3, 0.);
std::cout << B(0, 0);  // runtime assert error; B is immutable but T& operator()(unsigned, unsigned) is called.
B(0, 0) = 1.;  // same error

matrix C;
C = matrix(2, 3, 0.);
std::cout << C(0, 0);  // OK
C(0, 0) = 1.;  // OK
~~~
Be careful that as the declaration of `B` above shows, only `T& operator()(unsigned, unsigned)` is called if the matrix is not declared `const`. This is why `const` declaration is always recommended for immutable matrices, and I don't find any reason not to do.

### Defining your own function

Basic knowledge of copy/move-construct of C++11 is needed when passing matrices in and out of functions.

This `triple` is a working function for matrices. It is defined as a function template to accept any type `T`.
~~~C++
template <typename T>
matrix<T> triple(matrix<T> A)
{ return 3 * A; }
~~~
However, `triple` simply takes `matrix<T> A` as its argument, which copies all thunks and sub-thunks that `A` might have. Since we already know `A` is immutable (recall that all matrix variables with initialization such as function arguments are immutable), we can save the copying:
~~~C++
template <typename T>
matrix<T> triple(const matrix<T>& A)
{ return 3 * A; }
~~~
The above `triple` does not copy the argument into `A` but have `A` just refer to the argument, which is good for saving copying. However, the argument is assumed to lie in a matrix variable (as is suggested by the lvalue reference). If the argument comes from temporary value that will be destroyed soon, we don't need to preserve the argument. So, we can overload `triple` with another definition:
~~~C++
template <typename T>
matrix<T> triple(const matrix<T>& A)
{ return 3 * A; }
template <typename T>
matrix<T> triple(matrix<T>&& A)
{ return 3 * std::move(A); }  // std::move(A) is necessary to indicate A is an rvalue reference.
~~~

The `return 3 * A` and `return 3 * std::move(A)` makes an immutable matrix and returns it as an rvalue reference outside.

### Defining functions by inheriting `matrix<T>` class

The `triple` in above section can be also defined by inheriting `matrix<T>`:
~~~C++
template <typename T>
struct triple: matrix<T> {
    triple(const matrix<T>& A): matrix<T>(3 * A) {}
    triple(matrix<T>&& A): matrix<T>(3 * std::move(A)) {}
};
~~~
Note that `3 * A` and `3 * std::move(A)` computes and creates an immutable matrix as an rvalue reference, which is then passed to the move constructor of `matrix<T>`. The `operator*()` actually distinguishes lvalue reference and rvalue reference for its second argument, and if it is lvalue reference, `operator*()` copies thunk(s) from it to build the resulting matrix, whereas if rvalue reference, `operator*()` simply moves (i.e, recycles) those thunk(s).

`matrix<T>` has rich set of constructors besides copy and move constructors, which you can make use of to create your own functions. For example, `Id(n)` and `transpose(A)` functions are defined in this module as:
~~~C++
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
~~~

### Runtime checking of invalid matrix operations

Without defining compile constant `NDEBUG`, this module check for the validity of matrix operations such as sizes-matching on matrix addition/multiplication and range-checking on using subscript operator, `A(i, j)`.
