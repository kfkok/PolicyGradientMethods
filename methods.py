from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import tensorflow.keras.backend as K


# turn off eager execution
tf.compat.v1.disable_eager_execution()


def multiply_gradient(grad, multiplier):
    grad2 = []
    multiplier = multiplier.flatten()
    for i in range(len(grad)):
        grad2.append(grad[i] * multiplier)
    return grad2


class OneStepActorCritic():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.discount = 0.9

        # placeholders
        state = Input(batch_shape=(None, state_size), name="state")
        action_indice = K.placeholder(shape=1, dtype=tf.int32)

        # critic network
        hidden = Dense(30, activation='tanh', kernel_initializer='he_uniform')(state)
        output = Dense(1, activation='linear', kernel_initializer='he_uniform')(hidden)
        self.critic = Model(name='critic', inputs=state, outputs=output)
        self.critic.summary()
        self.critic_lr = 0.01
        critic_weights = self.critic.trainable_weights
        critic_loss = self.critic.output
        critic_gradient = K.gradients(loss=critic_loss, variables=critic_weights)

        # actor network
        actor_hidden = Dense(20, activation='tanh', kernel_initializer='he_uniform')(state)
        output = Dense(action_size, activation='softmax')(actor_hidden)
        self.actor = Model(name='actor', inputs=state, outputs=output)
        self.actor.summary()
        self.actor_lr = 0.01
        actor_weights = self.actor.trainable_weights
        ln_actor_output = K.log(self.actor.output)
        actor_loss = K.gather(K.reshape(ln_actor_output, [-1]), action_indice)
        actor_gradient = K.gradients(loss=actor_loss, variables=actor_weights)

        # A function to get the critic and actor gradients
        self.get_gradients = K.function(inputs=[state, action_indice], outputs=[critic_gradient, actor_gradient])

    def get_action_probabilities(self, state):
        state = state.reshape(1, self.state_size)
        return self.actor.predict(state)

    def compute_td_error(self, state, next_state, reward, done):
        next_state_value = self.critic.predict(next_state)
        state_value = self.critic.predict(state)
        target_value = reward
        if not done:
            target_value += self.discount * next_state_value
        td_error = target_value - state_value
        return td_error

    def train(self, state, action, next_state, reward, done, I):
        state = state.reshape(1, self.state_size)
        next_state = next_state.reshape(1, self.state_size)

        td_error = self.compute_td_error(state, next_state, reward, done)
        critic_gradient, actor_gradient = self.get_gradients([state, action])
        critic_weights = self.critic.get_weights()
        actor_weights = self.actor.get_weights()

        # compute the new critic weights
        critic_weights_delta = multiply_gradient(critic_gradient, self.critic_lr * td_error)
        new_critic_weights = [w + delta_w for w, delta_w in zip(critic_weights, critic_weights_delta)]

        # compute the new actor weights
        actor_weights_delta = multiply_gradient(actor_gradient, self.actor_lr * td_error)
        new_actor_weights = [w + delta_w for w, delta_w in zip(actor_weights, actor_weights_delta)]

        # print("td_error:", td_error)
        # print("actor_gradient:", actor_gradient[0][0][0:3])
        # print("actor_weights_delta:", actor_weights_delta[0][0][0:3])
        # print("new_actor_weights:", new_actor_weights[0][0][0:3])
        # print("")

        # update the weights for critic and actor
        self.critic.set_weights(new_critic_weights)
        self.actor.set_weights(new_actor_weights)

        return I * self.actor_lr
