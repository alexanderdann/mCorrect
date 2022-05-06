class TransformError(Exception):
    def __init__(self, nonlinearities, transformation, n_sets):
        self.message = f'Invalid combination of passed arguments: nonlinearities={nonlinearities}, transformation={transformation} and n_sets={n_sets}.'
        super().__init__(self.message)