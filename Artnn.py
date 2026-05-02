import numpy as np

# Parameters
num_features = 4
num_clusters = 3
vigilance = 0.6  # Similarity threshold

# Input Patterns (Binary)
X = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 0]
])

# Initialize Weights
# Bottom-up weights
W = np.ones((num_clusters, num_features)) / (1 + num_features)
# Top-down weights
T = np.ones((num_clusters, num_features))

# ART1 Algorithm
def art1(X, W, T, vigilance):
    clusters = []
    
    for i, x in enumerate(X):
        print(f"\nInput Pattern {i+1}: {x}")
        
        while True:
            # Compute matching scores
            net = np.dot(W, x)
            # Choose best matching neuron
            j = np.argmax(net)
            
            # Compute match (similarity)
            match = np.sum(np.minimum(x, T[j])) / np.sum(x)
            
            if match >= vigilance:
                print(f"Assigned to cluster {j}")
                # Update weights
                T[j] = np.minimum(x, T[j])
                W[j] = T[j] / (0.5 + np.sum(T[j]))
                clusters.append(j)
                break
            else:
                # Reset this neuron and try next
                W[j] = -1
                
                # Check if all neurons have been tried and failed
                if np.all(W == -1):
                    print("No suitable cluster found")
                    clusters.append(-1)
                    break
                    
    return clusters

# Run ART1
if __name__ == "__main__":
    # Use .copy() so we don't permanently alter the original weight matrices
    # if we decide to run the function multiple times in a larger script.
    cluster_result = art1(X, W.copy(), T.copy(), vigilance)
    print("\nFinal Cluster Assignments:", cluster_result)