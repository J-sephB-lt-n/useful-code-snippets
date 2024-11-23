"""
TAGS: common|dsa|leetcode|longest|longest common substring|substring
DESCRIPTION: Function for finding the length of the longest common substring between 2 strings
NOTES: I took this code from https://www.geeksforgeeks.org/longest-common-substring-dp-29/
"""

def len_of_longest_common_substring(s1: str, s2: str) -> int:
    """
Finds the length of the longest common substring between 2 strings 

Examples:
    >>> len_of_longest_common_substring("I wish that you hadn't told me that yesterday", "she told me that")
    13
    >>> len_of_longest_common_substring("they told me that too", "I wish that you hadn't told me that yesterday")
    14
    >>> len_of_longest_common_substring("qpalzmdifspnokwenfpmndsf", "pjvqpalzmqpalzm") # qpalzm
    6
    >>> len_of_longest_common_substring("1234567890", "0987654321")
    1
    >>> len_of_longest_common_substring("abc", "xyz")
    0
    """
    m = len(s1)
    n = len(s2)

    # Create a 1D array to store the previous row's results
    prev = [0] * (n + 1)
    
    res = 0
    for i in range(1, m + 1):
        # Create a temporary array to store the current row
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
                res = max(res, curr[j])
            else:
                curr[j] = 0
        
        # Move the current row's data to the previous row
        prev = curr
    
    return res

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
