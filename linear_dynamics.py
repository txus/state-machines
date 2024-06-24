import numpy as np

from state_machine import StateMachine

class LinearDynamicsSystem(StateMachine):
    def __init__(self, A, B, C):
        """
        Initializes the linear dynamics system with state transition, control matrices, and output matrix.

        :param A: np.array, the state transition matrix.
        :param B: np.array, the control input matrix.
        :param C: np.array, the output matrix. The output at each step is C * state.
        """
        self.A = A
        self.B = B
        self.C = C

    def create_initial_state(self):
        """Initialize the system's state as zeros."""
        return np.zeros((self.A.shape[0], 1))

    def create_example_inputs(self):
        return [
            np.array([0.1, 0.2, 0.3, 0.4]),
        ]

    def step(self, state, input):
        """
        Performs one time step update of the system's state and calculates the output.

        :param state: np.array, the current state of the system.
        :param input: np.array, the control input for the time step.
        :return: Tuple of (next_state, output), where output is C * state for the updated state.
        """
        next_state = np.dot(self.A, state) + np.dot(self.B, input)
        output = np.dot(self.C, next_state)
        return next_state, output

    def blockstep(self, state, block_of_inputs):
        """
        Attempts to simulate multiple time steps of the system's evolution using numpy operations.  Note: Due to sequential dependency, fully vectorized state updates are not feasible without iteration.

        :param state: np.array, the initial state of the system.
        :param block_of_inputs: np.array, a 2D array where each row is a control input for a time step.
        :return: Tuple of (next_state, outputs) where outputs include outputs at each step.
        """
        block_of_inputs = np.array(block_of_inputs)
        # Pre-compute all state updates (though still sequentially for accuracy)
        states = np.hstack(
            [state] + [state] * block_of_inputs.shape[1])  # Initialize placeholder for states
        for i in range(block_of_inputs.shape[1]):
            state = np.dot(self.A, state) + np.dot(self.B, block_of_inputs[:, i:i + 1])
            states[:, i + 1:i + 2] = state

        # Now, vectorize output calculation over all states
        outputs = np.dot(self.C, states)

        return states[:, -1:], outputs


# Example usage
if __name__ == '__main__':
    A = np.array([[1, 2], [0, 1]])
    B = np.array([[0.5], [1]])
    C = np.array([[1, 0], [0, 1]])
    lds = LinearDynamicsSystem(A, B, C)
    initial_state = lds.create_initial_state()
    inputs = lds.create_example_inputs()
    final_state, outputs = lds.blockstep(initial_state, inputs)
    print(f"Final State: {final_state}, Outputs: {outputs}")
