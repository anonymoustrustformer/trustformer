class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [0] * len(parameters)
        self.v = [0] * len(parameters)
        self.t = 0

    def step(self, gradients):
        self.t += 1
        for i in range(len(self.parameters)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients[i] ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            self.parameters[i] -= self.lr * m_hat / (v_hat ** 0.5 + self.epsilon)

    def zero_grad(self):
        # This function can be used to reset gradients if necessary
        pass
