import tensorflow as tf
import numpy as np

class RBM():
  """Restricted Boltzmann Machine."""
  def __init__(self, n_visible, n_hidden, cd_steps=3):
    self._n_visible = n_visible
    self._n_hidden = n_hidden
    self._weights = tf.Variable(tf.random.normal([self._n_visible, self._n_hidden]) * 0.01)
    self._bias_visible = tf.Variable(tf.zeros([self._n_visible, 1]))
    self._bias_hidden = tf.Variable(tf.zeros([self._n_hidden, 1]))
    self._cd_steps = cd_steps

  def sample_bernoulli(self, probabilities):
    """Sampling based on bernoulli distribution."""
    return tf.nn.relu(tf.sign(probabilities - tf.random.uniform(probabilities.shape)))

  def sample_gaussian(self, probabilities):
    """Sampling based on normal distribution."""
    return tf.add(probabilities, tf.random.normal(probabilities.shape, mean=0.0, stddev=1.0))

  def sample(self, probabilities):
    """Sampling visible or/and hidden units."""
    return self.sample_bernoulli(probabilities)

  def probabilities_hidden(self, visible):
    """Getting probabilities of hidden units given visible units."""
    probabilities_h_given_v = tf.nn.sigmoid(tf.matmul(visible, self._weights) + tf.squeeze(self._bias_hidden))
    return probabilities_h_given_v

  def probabilities_visible(self, hidden):
    """Getting probabilities of visible units given hidden units."""
    probabilities_v_given_h = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self._weights)) + tf.squeeze(self._bias_visible))
    return probabilities_v_given_h

  def gibbs_step(self, visible):
    """Sampling hidden state from visible state and reconstruct visible state from hidden state."""
    hidden_prob = self.probabilities_hidden(visible)
    hidden_state = self.sample(hidden_prob)
    visible_prob = self.probabilities_visible(hidden_state)
    visible_state = visible_prob
    return hidden_prob, hidden_state, visible_prob, visible_state

  def energy(self, visible):
    """Computing the energy of the model."""
    bias_term = tf.matmul(visible, self._bias_visible)
    linear_transform = tf.matmul(visible, self._weights) + tf.squeeze(self._bias_hidden)
    hidden_term = tf.reduce_sum(tf.math.log(1 + tf.exp(linear_transform)), axis=1)
    return tf.reduce_mean(-hidden_term - bias_term)

  def contrastive_divergence(self, visible, cd_steps=1):
    """Constrastive divergence to train the model."""
    hidden_prob, _, _, fake_v_state = self.gibbs_step(visible)
    pos_divergence = tf.matmul(tf.transpose(visible), hidden_prob)

    for _ in range (cd_steps-1):
      _, _, _, fake_v_state = self.gibbs_step(fake_v_state)

    fake_h_prob = self.probabilities_hidden(fake_v_state)
    fake_h_state = self.sample(fake_h_prob)

    neg_divergence = tf.matmul(tf.transpose(fake_v_state), fake_h_prob)    

    dW = pos_divergence - neg_divergence
    dvb = visible - fake_v_state
    dhb = hidden_prob - fake_h_prob

    loss = tf.reduce_mean(tf.math.squared_difference(visible, fake_v_state))
    return dW, dvb, dhb, loss

  def update(self, visible, learning_rate, cd_steps=1):
    """Updating trainable variables (i.e. weights, visible and hidden biases)."""
    batch_size = tf.cast(tf.shape(visible)[0], tf.float32)
    dW, dvb, dhb, loss = self.contrastive_divergence(visible, cd_steps)

    delta_w = learning_rate / batch_size * dW
    delta_vb = learning_rate / batch_size * (tf.reduce_sum(tf.transpose(dvb), 1, keepdims=True))
    delta_hb = learning_rate / batch_size * (tf.reduce_sum(tf.transpose(dhb), 1, keepdims=True))    

    self._weights = self._weights + delta_w
    self._bias_visible = self._bias_visible + delta_vb
    self._bias_hidden = self._bias_hidden + delta_hb

    return loss

  def fit(self, X, training_steps, learning_rate=0.01, batch_size=64, cd_steps=1, display_step=1000):
    """Fitting the model."""
    data_train = tf.data.Dataset.from_tensor_slices(X)
    data_train = data_train.repeat().shuffle(X.shape[0]).batch(batch_size).prefetch(1)

    self._optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    for step, X_batch in enumerate(data_train.take(training_steps + 1)):
      loss = self.update(X_batch, learning_rate, cd_steps=cd_steps)

      if step % display_step == 0:
        print('Step {}, loss: {}'.format(step, loss))