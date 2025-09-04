import random
import numpy as np
from scipy.optimize import curve_fit

def sample_dict(dictionary, sample_size=1, ordered=False):
    """
    Randomly samples a subset of aligned data from a dictionary of lists.
    
    Args:
        dictionary (dict): A dictionary where each value is a list of the same length.
        sample_size (int): Number of items to sample (default: 1).
    
    Returns:
        dict: A new dictionary with sampled data.
    """
    # Check if all lists have the same length
    lengths = [len(v) for v in dictionary.values()]
    #if not all(l == lengths[0] for l in lengths):
     #   raise UserWarning("Dictionaries have different lengths!")
    
    total_items = min(lengths)
    if sample_size > total_items:
        UserWarning (f"Sample size ({sample_size}) exceeds total items ({total_items})!")
        sample_size = total_items
    
    # Generate random indices (without replacement)
    sampled_indices = random.sample(range(total_items), sample_size)
    if ordered:
        sampled_indices.sort()
    
    # Extract sampled data for each key
    sampled_data = {
        key: [value[i] for i in sampled_indices]
        for key, value in dictionary.items()
    }
    
    return sampled_data


def merge_dict(dict1, dict2):
    """Merge dict2 into dict1 in-place. Returns 1 on success, 0 on error."""
    try:
        dict1.update(dict2)
        return 1
    except (TypeError, AttributeError):
        return 0
    

def print_dict(dict):
    for key, value in dict.items():
        print(f"{key}: {value}")


def remove_duplicates_paired(array1, array2):
    """
    Removes duplicates from paired arrays of mixed scalar/array elements
    Keeps first occurrence of duplicates in array1 and corresponding array2 elements
    
    Args:
        array1: Array with potential duplicates (list or NumPy array)
        array2: Paired array (same length)
    
    Returns:
        Tuple of deduplicated (array1, array2) in same format as input
    """
    # Convert to numpy arrays of objects to handle mixed types
    arr1 = np.asarray(array1, dtype=object)
    arr2 = np.asarray(array2, dtype=object)
    
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must be of the same length")
    
    seen = {}
    unique_mask = np.ones(len(arr1), dtype=bool)
    
    for i, item in enumerate(arr1):
        # Create a consistent hashable representation
        if isinstance(item, np.ndarray):
            # For arrays, use tuple representation (shape + flattened values)
            signature = (item.shape, item.tobytes())
        elif isinstance(item, (list, tuple)):
            # For sequences, convert to tuple
            signature = tuple(item)
        else:
            # For scalars, use as-is
            signature = item
        
        # Check if we've seen this signature before
        if signature in seen:
            unique_mask[i] = False
        else:
            seen[signature] = True  # Mark as seen
    
    return arr1[unique_mask], arr2[unique_mask]

def contains_vector(vector: np.ndarray, existing_points_set: set) -> bool:
    """
    Check if a vector exists in a set of precomputed points (O(1) lookup).
    
    Args:
        vector (np.ndarray): The vector to check (e.g., [x,y,z] as a numpy array).
        existing_points_set (set): Prebuilt set of tuples representing existing points.
    
    Returns:
        bool: True if the vector is found, False otherwise.
    """
    vector_tuple = tuple(vector.tolist())  # Convert numpy array to hashable tuple
    return vector_tuple in existing_points_set
    
import numpy as np

def find_collision(current_pos, direction, candidate_points, r):
    A = current_pos
    d = direction
    
    # Vectorized calculations
    Ap = A - candidate_points  # Shape: (n, 3)
    b = 2.0 * np.dot(Ap, d)    # Shape: (n,)
    c = np.sum(Ap**2, axis=1) - r*r  # Shape: (n,)
    
    # Calculate discriminant
    discriminant = b**2 - 4*c  # Shape: (n,)
    
    # Identify valid collisions (discriminant >= 0)
    valid_mask = discriminant >= 0
    if not np.any(valid_mask):
        return None, float('inf')
    
    # Filter valid candidates
    valid_b = b[valid_mask]
    valid_c = c[valid_mask]
    valid_discriminant = discriminant[valid_mask]
    valid_points = candidate_points[valid_mask]
    
    # Vectorized root calculations
    sqrt_disc = np.sqrt(valid_discriminant)
    t1 = (-valid_b - sqrt_disc) / 2
    t2 = (-valid_b + sqrt_disc) / 2
    
    # Find positive roots and determine minimum t
    positive_roots = np.where((t1 > 0) | (t2 > 0), 
                               np.where((t1 > 0) & (t2 > 0), 
                                        np.minimum(t1, t2),
                                        np.where(t1 > 0, t1, t2)),
                               np.inf)
    
    min_idx = np.argmin(positive_roots)
    min_t = positive_roots[min_idx]
    
    return valid_points[min_idx], min_t

def sigmoid(x, width_coef=1):
    return 1 / (1 + np.exp(-(8/width_coef)*(x))) # width_coef=1 means the width of the sigmoid effective range is 1

def get_uniqueness_score(input_vector, set_vectors):
    """
    Calculates the uniqueness score of an input vector relative to a set of vectors.
    
    Parameters:
    input_vector (np.ndarray): A 1D numpy array of shape (n) representing the input vector.
    set_vectors (np.ndarray): A 2D numpy array of shape (m, n) representing the set of vectors.
    
    Returns:
    float: The uniqueness score, which is a ratio adjusted by a penalty factor.
    """
    input_vector = np.asarray(input_vector)
    set_vectors = np.asarray(set_vectors)

    # Validate input dimensions
    assert input_vector.ndim == 1, "input_vector must be a 1D array"
    assert set_vectors.ndim == 2, "set_vectors must be a 2D array"
    assert input_vector.shape[0] == set_vectors.shape[1], "Feature dimension mismatch between input_vector and set_vectors"
    
    # Compute centroid of the set vectors
    centroid = np.mean(set_vectors, axis=0)
    
    # Calculate distances from each vector in the set to the centroid
    distances_to_centroid = np.linalg.norm(set_vectors - centroid, axis=1)
    avg_dist_centroid = np.mean(distances_to_centroid)
    
    # Handle case where all vectors in the set are nearly identical (zero diversity)
    epsilon = 1e-9
    if avg_dist_centroid < epsilon:
        # Compute distances from input to each vector in the set
        distances_to_input = np.linalg.norm(set_vectors - input_vector, axis=1)
        min_dist = np.min(distances_to_input)
        
        if min_dist < epsilon:
            # Input is nearly identical to the set vectors
            return 0.0
        else:
            # Input is distinct from the (nearly identical) set vectors
            return min_dist
    
    # Calculate distances from input vector to each vector in the set
    distances_to_input = np.linalg.norm(set_vectors - input_vector, axis=1)
    avg_dist_input = np.mean(distances_to_input)
    min_dist = np.min(distances_to_input)
    
    # Compute the ratio of min distance to average centroid distance (penalty factor base)
    r = min_dist / avg_dist_centroid
    penalty_factor = sigmoid(r-0.5, width_coef=0.1)  # Linear penalty factor
    
    # Compute the uniqueness score
    uniqueness_score = (avg_dist_input / avg_dist_centroid) * penalty_factor

    # print(f"distances_to_input: {avg_dist_input}, avg_dist_centroid: {avg_dist_centroid}")
    
    return uniqueness_score

def get_diffusion_probability(min_path_len, max_path_len, path_len):
    diff = max_path_len - min_path_len
    prob = -sigmoid(path_len-(min_path_len+diff/4), width_coef=diff)+1
    return prob




def fit_exponential_increment(signal):
    if len(signal) == 0:
        return (0.0, 0.0)
    
    t_data = np.arange(len(signal))
    
    if signal[0] < 1:
        initial_A0 = 1 - signal[0]
        if len(signal) >= 2 and signal[1] < 1:
            if (1 - signal[0]) > 1e-6 and (1 - signal[1]) > 1e-6:
                ratio = (1 - signal[1]) / (1 - signal[0])
                if ratio > 1e-6:
                    initial_k = -np.log(ratio)
                else:
                    initial_k = 10.0
            else:
                initial_k = 0.1
        else:
            initial_k = 0.1
    else:
        initial_A0 = 0.0
        initial_k = 0.0
    
    def model(t, A0, k):
        return 1 - A0 * np.exp(-k * t)
    
    lower_bounds = [0, 0]
    upper_bounds = [np.inf, np.inf]
    
    try:
        popt, _ = curve_fit(model, t_data, signal, p0=[initial_A0, initial_k], bounds=(lower_bounds, upper_bounds))
        A0_fit, k_fit = popt
    except RuntimeError:
        A0_fit, k_fit = initial_A0, initial_k
    
    return (A0_fit, k_fit)

import numpy as np

def get_fitted_curve(A0, k, N):
    """
    Generate the fitted exponential increment curve for N time points.
    
    Parameters:
    A0 (float): The amplitude parameter from the fitted model.
    k (float): The rate constant from the fitted model.
    N (int): Number of samples to generate (time points from 0 to N-1).
    
    Returns:
    numpy.ndarray: Array of N values representing the fitted curve.
    """
    t = np.arange(N)
    return 1 - A0 * np.exp(-k * t)


class FixedSizeArray:
    def __init__(self, max_size):
        if max_size < 0:
            raise ValueError("max_size must be >= 0")
        self.max_size = max_size
        self._data = [None] * max_size  # Fixed-size internal storage
        self._size = 0  # Current number of elements (can be < max_size)

    def append(self, item):
        if self._size >= self.max_size:
            raise IndexError(f"Cannot append: FixedSizeArray is full (max_size={self.max_size})")
        self._data[self._size] = item
        self._size += 1

    def pop(self):
        if self._size == 0:
            raise IndexError("pop from empty FixedSizeArray")
        self._size -= 1
        item = self._data[self._size]
        self._data[self._size] = None  # Optional cleanup
        return item

    def __getitem__(self, index):
        if isinstance(index, slice):
            # Handle slicing (optional but useful)
            return self._data[:self._size][index]
        if index < 0:
            index += self._size
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range for size {self._size}")
        return self._data[index]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise NotImplementedError("Slicing assignment not supported")
        if index < 0:
            index += self._size
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range for size {self._size}")
        self._data[index] = value

    def __len__(self):
        return self._size

    def __repr__(self):
        return f"FixedSizeArray({self._data[:self._size]}, max_size={self.max_size})"

    # Optional: support iteration
    def __iter__(self):
        return iter(self._data[:self._size])