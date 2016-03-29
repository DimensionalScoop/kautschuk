def generate_fibonacci_number(target_elem):
    previous_elem = 0
    current_elem = 1
    for i in range(target_elem):
        next_elem = current_elem + previous_elem
        previous_elem = current_elem
        current_elem = next_elem
    return current_elem

T_cold_bucket = 230  # K
phi_mirror = 1.764  # rad
distance_microphone = 10  # m
