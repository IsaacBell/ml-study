def conv2d(data, kernel):
    m, n = len(data), len(data[0])
    k = len(kernel)

    # assume that the input is valid otherwise assert res = []
    res = []

    for i in range(m - k + 1):
        row = []

        for j in range(n - k + 1):
            val = 0

            for p in range(k):
                for q in range(k):
                    val += data[i+p][j+q] * kernel[p][q]

            row.append(val)
        res.append(row[::])

    return res
