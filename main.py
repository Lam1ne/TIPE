from self_learning import AutoApprentissageLC0
import tensorflow as tf

def main():
    reseau = tf.keras.models.load_model("path_to_save_model.h5")
    reseau.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mse'},
                   metrics={'policy_output': 'accuracy', 'value_output': tf.keras.metrics.MeanSquaredError()})
    auto_apprentissage = AutoApprentissageLC0(reseau)
    auto_apprentissage.lancer_auto_apprentissage()

if __name__ == "__main__":
    main()