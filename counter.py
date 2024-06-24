import numpy as np

from state_machine import StateMachine


class Counter(StateMachine):
    """Counts the number of elements seen so far."""
    def create_initial_state(self):
        # Initial state is a single-element array with the initial count set to 0
        return np.array(0)

    def create_example_inputs(self):
        return [
            np.array([-1, 3]),
            np.array([1, 5]),
            np.array([5, 2])
        ]

    def step(self, state, input):
        # The tick method increases the count by the number of elements in the input array
        next_state = state + np.array(input.size)
        # For simplicity, the output is just the current count, same as the next state
        output = next_state.copy()
        return next_state, output

    def blockstep(self, state, block_of_inputs):
        # Calculate the total number of elements in each input
        elements_per_input = np.prod(block_of_inputs.shape[1:])

        # Calculate the cumulative sum of the total elements
        cumulative_elements = state + np.arange(len(block_of_inputs)) * elements_per_input

        # The next state is the last element in the cumulative sum array
        next_state = cumulative_elements[-1].copy()

        # The outputs are the cumulative sums themselves, adjusted for the initial state
        outputs = cumulative_elements

        return next_state, outputs

if __name__ == '__main__':
    counter = Counter()
    counter_state = counter.create_initial_state()
    print("Initial Counter State:", counter_state)
    for input in counter.create_example_inputs():
        counter_state, output = counter.step(counter_state, input)
        print(f"{input=} {counter_state=} {output=}")
