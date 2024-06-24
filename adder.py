import numpy as np

from state_machine import StateMachine


class Adder(StateMachine):
    def create_initial_state(self):
        # Initial state is a single-element array with the initial count set to 0
        return np.array(0.)

    def create_example_inputs(self):
        return [
            np.array([-1., 3.]),
            np.array([1., 5.]),
            np.array([5., 2.])
        ]

    def step(self, state, input):
        # The tick method increases the count based on the sum of the input array
        next_state = state + np.sum(input)
        # For simplicity, the output is just the next state
        output = next_state.copy()
        return next_state, output

    def blockstep(self, state, block_of_inputs):
        if len(block_of_inputs.shape) > 1:
            block_of_inputs = block_of_inputs.sum(tuple(range(1, len(block_of_inputs.shape))))

        # Calculate the cumulative sum of the inputs' sums
        cumulative_inputs = state + np.cumsum(block_of_inputs, axis=0)

        # The next state is the last element in the cumulative sum array
        next_state = cumulative_inputs[-1].copy()

        # The outputs are the cumulative sums themselves
        outputs = cumulative_inputs

        return next_state, outputs

if __name__ == '__main__':
    adder = Adder()
    adder_state = adder.create_initial_state()
    print("Initial Adder State:", adder_state)
    for input_value in adder.create_example_inputs():
        adder_state, output = adder.step(adder_state, input_value)
        print(f"{input_value=}, {adder_state=}, {output=}")
