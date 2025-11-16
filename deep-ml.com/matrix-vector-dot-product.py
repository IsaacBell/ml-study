# Return a list where each element is the dot product of a row of 'a' with 'b'.
# If the number of columns in 'a' does not match the length of 'b', return -1.

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	if (len(a) != len(b)):
        return -1

    out = []
    for i in range(len(a)):
        tot = sum(a[i][j] * b[j] for j in range(len(b)))
        out.append(tot)

    return out

