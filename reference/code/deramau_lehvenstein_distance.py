# Edit distance, with an additional step for factoring transposition into calculations

def Deramau(s: str, t: str) -> int:
	n, m = len(s), len(t)
    dp = [[0 for _ in range(m+1)] for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 1
            if s[i-1] == t[j-1]:
                cost = 0
                dp[i][j] = dp[i-1][j-1] 
            dp[i][j] = min(
                dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
            if (i > 1 and 
                j > 1 and 
                s[i-1] == t[j-2] and 
                s[i-2] == t[j-1]):
                dp[i][j] = min(dp[i][j],
                    dp[i-2][j-2] + 1)
    return dp[n][m]
