#include "matrix.hpp"

#if 0
int main()
{
    matrix A(3, 3, 0.);
    show(A, "A = ");

    matrix B = A;
    show(B, "B = ");

    matrix C = B;
    show(C, "C = ");

    matrix D;
    D = C;
    show(D, "D = ");

    matrix E = D;
    show(E, "E = ");
}
#endif

#if 0  // transpose()
template <typename T>
matrix<T> test(const matrix<T>& A)
{ return transpose(A); }

int main()
{
    matrix A(2, 3, 1.);
    show(A, "A = ");

    matrix tA = transpose(A);
    show(tA, "tA = ");

    matrix B(3, 2, 2.);
    show(B, "B = ");

    matrix tB = test(transpose(B));
    show(tB, "tB = ");
}
#endif

#if 0
template <typename T>
matrix<T> test(const matrix<T>& A)
{ return 2 * 3 * transpose(A); }

int main()
{
    matrix A = matrix(2, 3, 1.);
    show(A, "A = ");

    matrix B = test(A);
    show(B, "B = ");

    matrix C;
    C = B;
    show(C, "C = ");
}
#endif

#if 0  // exceptional cases
int main()
{
    //matrix<double> A = A;  // runtime error!
    //matrix<double> B = std::move(B);  // runtime error!
    matrix A = matrix(2, 3, 1.);
    matrix B;
    B = A;

    B = B;  // Ok
    show(B, "B = ");

    B = std::move(B);  // Ok
    show(B, "B = ");
}
#endif

#if 0  // map(f, A)
int main()
{
    matrix A = matrix(2, 3, 1.);
    //matrix B = map( [](double x){ return -x; }, A );  // that is, -A
    //Or, we can use:
    matrix B( [](double x){ return -x; }, A );

    show(B, "B = ");
}
#endif

#if 0
int main()
{
    matrix A = matrix(2, 3, [](unsigned i, unsigned j){ return 10.*i + j; });
    matrix B = matrix(3, 3, 1.);
    matrix I = Id(3);

    //const matrix C = 2 * B * I + I;
    const matrix C = 2 * A * B * I + A;
    std::cout << "C(0, 0) = " << C(0, 0) << "\n";
	// would cause an assert error without const.
    show(C, "C = ");
}
#endif

#if 1
template <typename T>
matrix<T> mutable_Id(unsigned size)
{
    matrix<T> Id;
    Id = matrix(size, size, [](unsigned i, unsigned j){ return T(i == j); });
    return std::move(Id);
}

int main()
{
    matrix A = mutable_Id<double>(3);
    show(A, "A = ");
}
#endif
