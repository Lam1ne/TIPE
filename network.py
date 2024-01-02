import tensorflow as tf
from keras import layers, models
import yaml

class ChessNetwork:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        # Load parameters from config
        input_shape = (8, 8, 112)
        num_filters = self.config['model']['filters']
        num_blocks = self.config['model']['residual_blocks']

        # Input layer
        inputs = layers.Input(shape=input_shape)

        # Initial convolution layer
        x = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')(inputs)

        # Repeated blocks with convolution, squeeze-and-excitation, and skip connections
        for _ in range(num_blocks):
            x = self.build_residual_block(x, num_filters)

        # Policy head
        policy_head = layers.Conv2D(filters=2, kernel_size=1, activation='relu')(x)
        policy_head = layers.Flatten()(policy_head)
        policy_output = layers.Dense(73 * 8 * 8, activation='softmax', name='policy_output')(policy_head)

        # Value head
        value_head = layers.Conv2D(filters=1, kernel_size=1, activation='relu')(x)
        value_head = layers.Flatten()(value_head)
        value_output = layers.Dense(1, activation='tanh', name='value_output')(value_head)  # win, draw, loss

        # Combine the heads
        outputs = [policy_output, value_output]

        # Create the model
        model = models.Model(inputs=inputs, outputs=outputs)

        return model

    def build_residual_block(self, x, num_filters):
        # Save the input value for the skip connection
        block_input = x

        # Convolutional layers with squeeze-and-excitation block in between
        x = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
        x = self.squeeze_and_excitation(x, num_filters)

        # Skip connection and ReLU activation
        x = layers.add([block_input, x])
        x = layers.ReLU()(x)

        return x

    def squeeze_and_excitation(self, input_tensor, num_filters, ratio=16):
        # Squeeze operation
        se = layers.GlobalAveragePooling2D()(input_tensor)
        se = layers.Reshape((1, 1, num_filters))(se)
        se = layers.Dense(num_filters // ratio, activation='relu')(se)
        se = layers.Dense(num_filters, activation='sigmoid')(se)

        # Excitation operation
        x = layers.Multiply()([input_tensor, se])
        return x

    def compile_model(self):
        learning_rate = self.config['training']['lr_values'][0]
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss={
                               'policy_output': 'categorical_crossentropy',
                               'value_output': 'mean_squared_error'
                           },
                           metrics={
                               'policy_output': 'accuracy',
                               'value_output': 'mean_squared_error'
                           })

    def train(self, data, labels):
        batch_size = self.config['training']['batch_size']
        epochs = self.config['training']['total_steps']
        return self.model.fit(data, labels, batch_size=batch_size, epochs=epochs)

    def predict(self, board_state):
        return self.model.predict(board_state)

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = tf.keras.models.load_model(file_path)

    def adjust_learning_rate(self):
        new_learning_rate = self.config['training']['lr_values'][0]
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=new_learning_rate),
                           loss={
                               'policy_output': 'categorical_crossentropy',
                               'value_output': 'mean_squared_error'
                           },
                           metrics={
                               'policy_output': 'accuracy',
                               'value_output': 'mean_squared_error'
                           })

# Load config from YAML file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create and train model
network = ChessNetwork(config)