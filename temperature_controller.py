import numpy as np

from state_machine import StateMachine


class TemperatureController(StateMachine):
    HEATING = 0
    COOLING = 1
    IDLE = 2

    def create_initial_state(self):
        # Initial state: [current temperature, mode]
        # Starting at 20 degrees Celsius and idle mode
        return {'temp': np.array([20.0]), 'mode': np.array([self.IDLE])}

    def create_example_inputs(self):
        return [
            np.array([22, 0.2])
        ]

    def step(self, state, input):
        desired_temp, external_influence = input
        current_temp = state['temp'][0]
        mode = state['mode'][0]

        # Adjust temperature based on mode
        if mode == self.HEATING:
            current_temp += 1.0  # Heating increases temp by 1 degree
        elif mode == self.COOLING:
            current_temp -= 1.0  # Cooling decreases temp by 1 degree

        # Apply external temperature influence
        current_temp += external_influence

        # Determine next mode based on desired temp
        if current_temp < desired_temp - 0.5:
            mode = self.HEATING
        elif current_temp > desired_temp + 0.5:
            mode = self.COOLING
        else:
            mode = self.IDLE

        next_state = {'temp': np.array([current_temp]), 'mode': np.array([mode])}
        output = mode
        return next_state, output

    def blockstep(self, state, block_of_inputs):
        next_states = []
        for input in block_of_inputs:
            state, output = self.step(state, input)
            next_states.append(output)
        # The final state after processing all inputs
        final_state = state
        # Outputs are all the intermediate states
        outputs = np.array(next_states)
        return final_state, outputs

if __name__ == '__main__':
    temperature_controller = TemperatureController()
    temp_controller_state = temperature_controller.create_initial_state()
    print("Initial TemperatureController State:", temp_controller_state)
    input = temperature_controller.create_example_inputs()[0]
    temp_controller_state, output = temperature_controller.step(temp_controller_state, input)
    print(f"{temp_controller_state=} {output=}")
