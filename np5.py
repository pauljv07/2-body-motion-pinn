import numpy as np
from scipy.integrate import solve_ivp
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("Using CPU. No GPU found.")

# Gravitational constant and mass of Body A
G = 1  # Gravitational Constant
M = 1  # Mass of A

# ODE definition
def derivatives(t, y):
    r, theta, vr, vtheta = y
    drdt = vr
    dthetadt = vtheta
    dvrdt = r * vtheta**2 - G * M / r**2
    dvthetadt = -2 * vr * vtheta / r
    return [drdt, dthetadt, dvrdt, dvthetadt]

# Generate training data
def generate_training_data(num_samples=1000, t_span=(0, 10), t_eval_points=1000):
    X_train = []
    y_train = []
    t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)

    for _ in range(num_samples):
        # Random initial conditions
        r0 = 1
        theta0 = np.random.uniform(0, 2*np.pi)
        vr0 = np.random.uniform(-1, 1)
        vtheta0 = np.random.uniform(0.5, 2.0)

        # Initial state vector
        y0 = [r0, theta0, vr0, vtheta0]

        # Solve the ODE
        sol = solve_ivp(derivatives, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

        # Use the initial condition and future positions as input and output
        X_train.append(y0)
        y_train.append(sol.y[:2, :].flatten())  # r and theta values over time

    return np.array(X_train), np.array(y_train)

# Generate the training data
X_train, y_train = generate_training_data()

# Define the neural network model
model = models.Sequential([
    layers.Input(shape=(4,)),  # 4 inputs: r, theta, vr, vtheta
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(y_train.shape[1], activation='linear')  # Output shape matches flattened r, theta over time
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

# Save the trained model in TensorFlow format
model.save("./trained_PINN_model_orbit", save_format='tf')

# Plot the loss history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss History')
plt.savefig("loss_history_plot_Orbit")


# Evaluate the model on new initial conditions
def evaluate_model(model, r0, theta0, vr0, vtheta0, t_span=(0, 10), t_eval_points=1000):
    t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)
    y0 = np.array([r0, theta0, vr0, vtheta0])

    # Predict future motion
    y_pred = model.predict(y0.reshape(1, -1)).reshape(2, -1)  # Reshape to (2, t_eval_points)
    r_pred, theta_pred = y_pred

    return r_pred, theta_pred, t_eval

# Test the model with new initial conditions
r0_test, theta0_test, vr0_test, vtheta0_test = 1.0, 0.0, 0.0, 1.0
r_pred, theta_pred, t_eval = evaluate_model(model, r0_test, theta0_test, vr0_test, vtheta0_test)

# Set up the figure for animation
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_ylim(0, 2)  # Set appropriate limits for r
line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    if frame < len(r_pred):
        line.set_data(theta_pred[:frame], r_pred[:frame])
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=30)

# Display the animation
plt.show()
