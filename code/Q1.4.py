Q1_4 = True

def function_f(k, N, sigma):
    value = 4/(3*((N-1)**2))*(((k-1)**2)/2 + (k-1))**2 + (sigma**2)/k
    return value

# Question 1.4
if Q1_4:
    for N in [25,50]:
        for sigma in [0.0, 0.1, 0.2]:
            min_value = float('inf')
            best_k = -1
            for k_prime in range(int((N-1)/2)):
                k = 2*k_prime + 1
                value = function_f(k, N, sigma)
                if value < min_value:
                    min_value = value
                    best_k = k
            print(f"N={N}, sigma={sigma} => k*={best_k}")

# Question 1.5
else:
    for N in [50, 100, 500]:
        for sigma in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            min_value = float('inf')
            best_k = -1
            for k_prime in range(int((N-1)/2)):
                k = 2*k_prime + 1
                value = function_f(k, N, sigma)
                if value < min_value:
                    min_value = value
                    best_k = k
            print(f"N={N}, sigma={sigma} => k*={best_k}")

