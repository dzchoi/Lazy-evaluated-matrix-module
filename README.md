# Lazy-evaluated matrix module in C++

This lazy-evaluated matrix module is intended to be used for calculations using matrices and writing functions that include matrix computations.

This module is written in C++17, mostly in C++11, to be user-friendly in terms of its usage, and at the same time to be highly efficient in memory handling. You need to have basic knowledge of C++11 such as lambda function, copy-construct, move-construct, copy-assignment, move-assignment, lvalue references and rvalue references, in order to use this module effectively, and in particular to write your own functions.

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
***From now on, we call:***
* immutable matrix: lazy-evaluated matrix that is bound to a variable or not.
* mutable matrix: eager-evaluated matrix that is always bound to a variable.
---
