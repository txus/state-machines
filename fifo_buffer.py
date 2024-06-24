import numpy as np

from state_machine import StateMachine


class FIFOBuffer(StateMachine):
    def __init__(self, capacity):
        """
        Initializes the FIFO buffer with a given capacity.
        """
        self.capacity = capacity
        self.buffer = np.full((capacity, 1),
                              np.nan)  # Using nan to indicate empty slots
        self.size = 0

    def create_initial_state(self):
        """
        Return the initial state of the buffer.
        """
        return self.buffer, self.size

    def create_example_inputs(self):
        return [
            1,
            2,
            3,
            4,
            5,
            6,
            7
        ]

    def step(self, state, input):
        """
        Not changed, remains for single element handling.
        """
        buffer, size = state
        output = None
        if size >= self.capacity:
            output = buffer[0]
            buffer = np.roll(buffer, -1)
            buffer[-1] = input
        else:
            buffer[size] = input
            size += 1
        return (buffer, size), output

    def blockstep(self, state, block_of_inputs):
        """
        Efficiently simulates adding multiple elements to the buffer and handles overflows.

        :param state: tuple, the initial state of the buffer and its current size.
        :param block_of_inputs: np.array, elements to be added to the buffer.
        :return: Tuple of (next_state, removed_elements), where next_state is the updated state of the buffer,
                 and removed_elements is a numpy array of elements removed due to overflow.
        """
        buffer, size = state
        block_of_inputs = np.array(block_of_inputs).reshape(-1, 1)
        input_length = block_of_inputs.shape[0]
        total_length = size + input_length

        # Determine if there's an overflow and calculate new size after addition
        if total_length <= self.capacity:
            # No overflow, just append inputs to the end of the buffer
            buffer[size:total_length, :] = block_of_inputs
            new_size = total_length
            removed_elements = np.array([])
        else:
            # Overflow occurs; calculate how many new elements can fit and which are removed
            overflow_count = total_length - self.capacity
            removed_elements = buffer[:min(size, overflow_count),
                               :].copy()  # Elements that will be removed

            # Move the surviving elements to the start of the buffer if there's any overflow
            if size > overflow_count:
                buffer[:size - overflow_count, :] = buffer[overflow_count:size, :]

            # Insert new inputs into the buffer, some may replace old elements
            new_inputs_start = max(0, input_length - self.capacity)
            buffer[-input_length:] = block_of_inputs[new_inputs_start:, :]
            new_size = self.capacity

        return (buffer, new_size), removed_elements

if __name__ == '__main__':
    capacity = 5
    fifo_buffer = FIFOBuffer(capacity)

    # Initial state of the buffer
    initial_state = fifo_buffer.create_initial_state()
    print("Initial state (buffer content):", initial_state[0].T, "Size:",
          initial_state[1])

    # Batch addition 1
    inputs = fifo_buffer.create_example_inputs()[:3]
    state_after_first_addition, outputs = fifo_buffer.blockstep(initial_state, inputs)
    print("\nState after first addition (buffer content):",
          state_after_first_addition[0].T, "Size:", state_after_first_addition[1])
    print("Outputs:", outputs.T)

    # Batch addition 2 (with potential overflow)
    inputs = fifo_buffer.create_example_inputs()[3:]
    final_state, outputs = fifo_buffer.blockstep(state_after_first_addition, inputs)
    print("\nFinal state (buffer content):", final_state[0].T, "Size:",
          final_state[1])
    print("Outputs:", outputs.T)
