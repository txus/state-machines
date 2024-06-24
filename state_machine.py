"""Base class for generic state machine."""
import abc


class StateMachine(abc.ABC):
    """A deterministic Markov process.

    The mathematical definition is as follows. The process begins at some initial state
    s_0. From there, it evolves according to some rule:
            s_{i+1} = f(s_i, x_i)
    Additionally, it gives an output at each moment:
            y_i = g(s_i, x_i)
    We combine these two into a "step" function:
            s_{i+1}, y_i = step(s_i, x_i) = ( f(s_i, x_i), g(s_i, x_i) )

    This class is a functional-style implementation of a state machine. To get s0, we
    use `SM.create_initial_state()`. The evolution of the process is determined by the
    `.step` function, which does: `s_{i+1}, y_i = SM.step(s_i, x_i)`.

    For computational efficiency, it can be helpful to compute several updates at once,
    rather than one at a time. To support this, the class has a second function,
    `.blockstep`, which is exactly equivalent to taking multiple steps:
        `s_j, y_i_to_j = SM.step(s_i, x_i_to_j)`
    As a result, the inputs and outputs to `.blockstep` have an additional leading
    dimension relative to that of `.step`. So, if we might call:
        `SM.step(state, input)` on `input.shape == [3, 5]`
    then we would call
        `SM.blockstep(state, block_of_inputs)` on `block_of_inputs.shape == [10, 3, 5]`
    In this case, it would return a block_of_outputs with leading dimension 10 as well.
    (Of course, block sizes other than 10 are allowed, too.)
    """

    @abc.abstractmethod
    def create_initial_state(self):
        """
        Returns an initial state.

        This should be a pytree: either an np array, or nested dicts/lists/tuples with
        np arrays at the leaves.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_example_inputs(self):
        """
        Creates a list of example inputs, to facilitate testing.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, state, input):
        """
        Executes a minimal unit of evolution of the state machine.

        In:
          state: a state, as returned by create_initial_state().
          input: a single input, which is a np array of a particular shape and dtype.
        Out:
          next_state: the state that results from an application of the process.
          output: the output at this moment of the process, which is a np array.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def blockstep(self, state, block_of_inputs):
        """
        Executes several steps of the process, all at once.

        In:
          state: a state, corresponding to the final state after applying all ticks.
          block_of_inputs: an array of inputs, whose shape should be like that of step,
                           but with an extra leading dimension.
        Out:
          next_state: the state that results from an application of the process.
          block_of_outputs: the output at each moment of the process, which will be an
                            array whose leading dimension matches that of inputs.
        """
        raise NotImplementedError

