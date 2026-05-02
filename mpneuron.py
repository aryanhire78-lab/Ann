# McCulloch-Pitts Neuron for AND-NOT Function
def mp_neuron(A, B):
    w1 = 1   # weight for A
    w2 = -1  # weight for B (NOT operation)
    threshold = 1
    
    net_input = A * w1 + B * w2
    
    if net_input >= threshold:
        return 1
    else:
        return 0

# Testing all input combinations
if __name__ == "__main__":
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    print("A B | Output (A AND NOT B)")
    print("-" * 26)
    
    for A, B in inputs:
        output = mp_neuron(A, B)
        # Using formatted strings for a cleaner output display
        print(f"{A} {B} | {output}")