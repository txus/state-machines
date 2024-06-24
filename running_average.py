import numpy as np

from state_machine import StateMachine


class RunningAverage(StateMachine):
    def create_initial_state(self):
        # Initial state includes the sum of inputs and count of inputs processed
        return {'total_sum': np.array([0.0]), 'count': np.array([0])}

    def create_example_inputs(self):
        return [
            10,
            20,
            30,
            40
        ]

    def step(self, state, input):
        # Updates the state with a new input and calculates the running average
        next_state = {
            'total_sum': state['total_sum'] + input,
            'count': state['count'] + 1
        }
        running_average = next_state['total_sum'] / next_state['count']
        output = np.array([running_average])
        return next_state, output

    def blockstep(self, state, block_of_inputs):
        # Efficiently processes multiple inputs to calculate the running average
        total_input_sum = np.sum(block_of_inputs)
        num_inputs = len(block_of_inputs)

        next_state = {
            'total_sum': state['total_sum'] + total_input_sum,
            'count': state['count'] + num_inputs
        }

        # Calculate running averages after each input in an efficient manner
        cumulative_sums = np.cumsum(block_of_inputs) + state['total_sum']
        running_averages = cumulative_sums / (num_inputs + state['count'])

        outputs = running_averages

        return next_state, outputs


if __name__ == '__main__':
    avg_calculator = RunningAverage()
    initial_state = avg_calculator.create_initial_state()
    print("Initial State:", initial_state)

    inputs = avg_calculator.create_example_inputs()
    final_state, running_averages = avg_calculator.blockstep(initial_state, inputs)

    print("Final State after processing inputs:", final_state)
    print("Running averages after each input:", running_averages)
