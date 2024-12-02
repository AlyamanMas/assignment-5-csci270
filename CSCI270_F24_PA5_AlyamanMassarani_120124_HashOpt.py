import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HashTable:
    def __init__(self, size, load_factor_threshold=0.7):
        self.size = size
        self.load_factor_threshold = load_factor_threshold
        self.table = [None] * self.size
        self.num_elements = 0
        self.original_elements = []  # Store original insertions

    def hash_function(self, x, a, b, m):
        return math.floor((a * x + b) % m)

    def insert(self, key, value, a, b):
        # Store original insertion for later use
        if (key, value) not in self.original_elements:
            self.original_elements.append((key, value))

        # Calculate load factor and rehash if necessary
        if self.num_elements / self.size >= self.load_factor_threshold:
            self.rehash(a, b)

        # Calculate hash index
        hash_index = self.hash_function(key, a, b, self.size)

        # Linear probing to handle collisions
        original_index = hash_index
        while self.table[hash_index] is not None:
            hash_index = (hash_index + 1) % self.size
            if hash_index == original_index:
                self.rehash(a, b)
                return self.insert(key, value, a, b)

        self.table[hash_index] = (key, value)
        self.num_elements += 1

    def rehash(self, a, b):
        old_table = self.table
        self.size = self.size * 2
        self.table = [None] * self.size
        self.num_elements = 0

        for item in old_table:
            if item is not None:
                key, value = item
                self.insert(key, value, a, b)

    def calculate_collisions(self, a, b):
        collisions = 0
        positions_used = set()

        for item in self.table:
            if item is not None:
                key, _ = item
                original_hash = self.hash_function(key, a, b, self.size)

                if original_hash in positions_used:
                    collisions += 1
                positions_used.add(original_hash)

        return collisions


def create_new_table_with_params(original_table, a, b, m):
    """Create a new hash table with given parameters and copy original elements."""
    new_table = HashTable(m)

    # Copy all original elements to new table
    for key, value in original_table.original_elements:
        new_table.insert(key, value, a, b)

    return new_table


def hill_climbing(original_table, initial_a, initial_b, initial_m, iterations=1000):
    current_a = initial_a
    current_b = initial_b
    current_m = initial_m

    # Create initial table with current parameters
    current_table = create_new_table_with_params(
        original_table, current_a, current_b, current_m
    )
    current_collisions = current_table.calculate_collisions(current_a, current_b)

    best_a, best_b, best_m = current_a, current_b, current_m
    best_collisions = current_collisions

    step_size_params = 1
    step_size_m = max(
        2, current_m // 10
    )  # Step size for m should be integer and scale with m

    for _ in range(iterations):
        # Try adjusting all parameters
        neighbors = [
            (current_a + step_size_params, current_b, current_m),
            (current_a - step_size_params, current_b, current_m),
            (current_a, current_b + step_size_params, current_m),
            (current_a, current_b - step_size_params, current_m),
            (current_a, current_b, current_m + step_size_m),
            (current_a, current_b, current_m - step_size_m),
        ]

        # Evaluate each neighbor
        best_neighbor_collisions = float("inf")
        best_neighbor = None

        for neighbor in neighbors:
            a, b, m = neighbor
            if m <= 0:  # Skip if m is too small
                continue

            # Create new table with neighbor parameters
            test_table = create_new_table_with_params(original_table, a, b, m)
            collisions = test_table.calculate_collisions(a, b)

            if collisions < best_neighbor_collisions:
                best_neighbor_collisions = collisions
                best_neighbor = neighbor

        # If no improvement, reduce step sizes
        if best_neighbor is None or best_neighbor_collisions >= current_collisions:
            step_size_params *= 0.95
            step_size_m = max(1, int(step_size_m * 0.95))
            if step_size_params < 0.0001 and step_size_m <= 1:
                break
        else:
            current_a, current_b, current_m = best_neighbor
            current_collisions = best_neighbor_collisions

            # Update best solution if current is better
            if current_collisions < best_collisions:
                best_a, best_b, best_m = current_a, current_b, current_m
                best_collisions = current_collisions

    return best_a, best_b, best_m, best_collisions


def simulated_annealing(
    original_table,
    initial_a,
    initial_b,
    initial_m,
    initial_temperature=1000,
    cooling_rate=0.003,
):
    current_a = initial_a
    current_b = initial_b
    current_m = initial_m

    # Create initial table with current parameters
    current_table = create_new_table_with_params(
        original_table, current_a, current_b, current_m
    )
    current_collisions = current_table.calculate_collisions(current_a, current_b)

    best_a, best_b, best_m = current_a, current_b, current_m
    best_collisions = current_collisions
    temperature = initial_temperature

    while temperature > 1:
        # Generate neighbor with random perturbation
        new_a = current_a + np.random.uniform(-0.5, 0.5)
        new_b = current_b + np.random.uniform(-0.5, 0.5)
        new_m = max(
            len(original_table.original_elements),
            int(current_m + np.random.randint(-5, 6)),
        )  # Ensure m stays reasonable

        # Create new table with new parameters
        new_table = create_new_table_with_params(original_table, new_a, new_b, new_m)
        new_collisions = new_table.calculate_collisions(new_a, new_b)

        # Calculate energy difference
        delta_e = new_collisions - current_collisions

        # Accept the new solution if it's better or with a probability
        if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
            current_a, current_b, current_m = new_a, new_b, new_m
            current_collisions = new_collisions

            # Update best solution if current is better
            if current_collisions < best_collisions:
                best_a, best_b, best_m = current_a, current_b, current_m
                best_collisions = current_collisions

        temperature *= 1 - cooling_rate

    return best_a, best_b, best_m, best_collisions


def nelder_mead(
    original_table,
    initial_a,
    initial_b,
    initial_m,
    alpha=1.0,
    beta=0.5,
    gamma=2.0,
    max_iter=1000,
):
    # Helper functions
    def centroid(simplex):
        """Calculate centroid of all points except the worst one"""
        return np.mean(simplex[:-1], axis=0)

    def reflect(simplex, centroid_point, alpha):
        """Reflect worst point through centroid"""
        worst = simplex[-1]
        reflected = centroid_point + alpha * (centroid_point - worst)
        # Ensure m stays integer and reasonable
        reflected[2] = max(len(original_table.original_elements), int(reflected[2]))
        return reflected

    def expand(simplex, centroid_point, reflected_point, gamma):
        """Expand reflected point further from centroid"""
        expanded = centroid_point + gamma * (reflected_point - centroid_point)
        # Ensure m stays integer and reasonable
        expanded[2] = max(len(original_table.original_elements), int(expanded[2]))
        return expanded

    def contract(simplex, centroid_point, beta):
        """Contract worst point toward centroid"""
        worst = simplex[-1]
        contracted = centroid_point + beta * (worst - centroid_point)
        # Ensure m stays integer and reasonable
        contracted[2] = max(len(original_table.original_elements), int(contracted[2]))
        return contracted

    def evaluate_point(point):
        """Calculate collisions for given parameters"""
        a, b, m = point
        table = create_new_table_with_params(original_table, a, b, int(m))
        return table.calculate_collisions(a, b)

    # Initialize simplex with four points (3D space: a, b, m)
    p1 = np.array([initial_a, initial_b, initial_m])
    p2 = np.array([initial_a + 1.0, initial_b, initial_m])
    p3 = np.array([initial_a, initial_b + 1.0, initial_m])
    p4 = np.array([initial_a, initial_b, initial_m + 5])

    # Create simplex and sort by function value
    simplex = np.array([p1, p2, p3, p4])
    values = np.array([evaluate_point(p) for p in simplex])
    order = np.argsort(values)
    simplex = simplex[order]
    values = values[order]

    best_point = simplex[0]
    best_value = values[0]

    # Main optimization loop
    for _ in range(max_iter):
        # Sort simplex points by function value
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        # Calculate centroid of best points
        centroid_point = centroid(simplex)

        # Reflect worst point
        reflected_point = reflect(simplex, centroid_point, alpha)
        reflected_value = evaluate_point(reflected_point)

        if reflected_value < values[0]:
            # If reflection is best so far, try expansion
            expanded_point = expand(simplex, centroid_point, reflected_point, gamma)
            expanded_value = evaluate_point(expanded_point)

            if expanded_value < reflected_value:
                simplex[-1] = expanded_point
                values[-1] = expanded_value
            else:
                simplex[-1] = reflected_point
                values[-1] = reflected_value

        elif reflected_value < values[-2]:
            # If reflection is better than second worst, accept it
            simplex[-1] = reflected_point
            values[-1] = reflected_value

        else:
            # Try contraction
            contracted_point = contract(simplex, centroid_point, beta)
            contracted_value = evaluate_point(contracted_point)

            if contracted_value < values[-1]:
                simplex[-1] = contracted_point
                values[-1] = contracted_value
            else:
                # Shrink all points toward best point
                best = simplex[0]
                for i in range(1, len(simplex)):
                    simplex[i] = best + beta * (simplex[i] - best)
                    simplex[i][2] = max(
                        len(original_table.original_elements), int(simplex[i][2])
                    )
                    values[i] = evaluate_point(simplex[i])

        # Update best solution if improved
        if values[0] < best_value:
            best_point = simplex[0]
            best_value = values[0]

        # Check for convergence
        if np.max(np.abs(values - values[0])) < 1e-6:
            break

    return best_point[0], best_point[1], int(best_point[2]), int(best_value)


def compare_iteration():
    # Create an instance of HashTable with initial size
    initial_size = 20
    hash_table = HashTable(initial_size)

    # Insert random keys into the hash table for testing
    # print("Inserting test data...")
    for i in range(50):
        hash_table.insert(random.randint(0, 1000), i, a=1, b=1)

    # Initial parameters
    initial_a = 1
    initial_b = 1
    initial_m = initial_size

    # print("\nOptimizing hash function parameters...")

    # Run Hill Climbing
    hc_a, hc_b, hc_m, hc_collisions = hill_climbing(
        hash_table, initial_a, initial_b, initial_m
    )
    # print(f"\nHill Climbing Results:")
    # print(f"Optimized parameters: a={hc_a:.4f}, b={hc_b:.4f}, m={hc_m}")
    # print(f"Final collision count: {hc_collisions}")

    # Run Simulated Annealing
    sa_a, sa_b, sa_m, sa_collisions = simulated_annealing(
        hash_table, initial_a, initial_b, initial_m
    )
    # print(f"\nSimulated Annealing Results:")
    # print(f"Optimized parameters: a={sa_a:.4f}, b={sa_b:.4f}, m={sa_m}")
    # print(f"Final collision count: {sa_collisions}")

    # Run Nelder-Mead
    nm_a, nm_b, nm_m, nm_collisions = nelder_mead(
        hash_table, initial_a, initial_b, initial_m
    )
    # print(f"\nNelder-Mead Results:")
    # print(f"Optimized parameters: a={nm_a:.4f}, b={nm_b:.4f}, m={nm_m}")
    # print(f"Final collision count: {nm_collisions}")

    # Compare results
    # print("\nComparison of Optimization Methods:")
    methods = {
        "Hill Climbing": (hc_collisions, hc_a, hc_b, hc_m),
        "Simulated Annealing": (sa_collisions, sa_a, sa_b, sa_m),
        "Nelder-Mead": (nm_collisions, nm_a, nm_b, nm_m),
    }

    best_method = min(methods.items(), key=lambda x: x[1][0])
    # print(f"\nBest performing method: {best_method[0]}")
    # print(
    #     f"Parameters: a={best_method[1][1]:.4f}, b={best_method[1][2]:.4f}, m={best_method[1][3]}"
    # )
    # print(f"Collision count: {best_method[1][0]}")
    return hc_collisions, sa_collisions, nm_collisions, best_method[0]


def main():
    # Run experiments
    num_iterations = 50
    results_df = []

    print("Running optimization experiments...")
    for i in range(num_iterations):
        if i % 10 == 0:
            print(f"Progress: {i}/{num_iterations}")
        results_df.append(compare_iteration())

    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results_df,
        columns=[
            "Hill Climbing Collisions",
            "Simulated Annealing Collisions",
            "Nelder-Mead Collisions",
            "Best method",
        ],
    )

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Bar chart of best performing methods
    best_counts = results_df["Best method"].value_counts()
    colors = ["#2ecc71", "#e74c3c", "#3498db"]  # Green, Red, Blue
    best_counts.plot(kind="bar", ax=ax1, color=colors)
    ax1.set_title("Best Performing Algorithm Distribution", pad=20)
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Number of Times Best")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels on top of each bar
    for i, v in enumerate(best_counts):
        ax1.text(i, v, str(v), ha="center", va="bottom")

    # Plot 2: Box plot of collisions
    collision_data = [
        results_df["Hill Climbing Collisions"],
        results_df["Simulated Annealing Collisions"],
        results_df["Nelder-Mead Collisions"],
    ]
    labels = ["Hill\nClimbing", "Simulated\nAnnealing", "Nelder-\nMead"]

    bp = ax2.boxplot(
        collision_data,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="gray"),
    )

    # Set colors for boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_title("Distribution of Collisions by Algorithm", pad=20)
    ax2.set_ylabel("Number of Collisions")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout and display
    plt.tight_layout()

    # Add a main title to the figure
    fig.suptitle("Hash Function Optimization Algorithm Comparison", fontsize=14, y=1.05)

    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nMean Collisions:")
    means = results_df[
        [
            "Hill Climbing Collisions",
            "Simulated Annealing Collisions",
            "Nelder-Mead Collisions",
        ]
    ].mean()
    print(means)

    print("\nMedian Collisions:")
    medians = results_df[
        [
            "Hill Climbing Collisions",
            "Simulated Annealing Collisions",
            "Nelder-Mead Collisions",
        ]
    ].median()
    print(medians)

    print("\nBest Algorithm Distribution:")
    distribution = results_df["Best method"].value_counts(normalize=True)
    print(distribution)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
