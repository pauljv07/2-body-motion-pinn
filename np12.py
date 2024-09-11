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

class CustomLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.loss_r = []
        self.loss_theta = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Predict using the model
        y_pred = self.model.predict(X_train, batch_size=5000)  # Adjust batch_size if needed
        
        # Calculate losses for r and theta
        loss_r = np.mean(np.square(y_train[:, 0] - y_pred[:, 0]))
        loss_theta = np.mean(np.square(y_train[:, 1] - y_pred[:, 1]))
        
        # Append losses to the lists
        self.loss_r.append(loss_r)
        self.loss_theta.append(loss_theta)
        
    def on_train_end(self, logs=None):
        # Finalize the list of losses
        self.loss_r = self.loss_r
        self.loss_theta = self.loss_theta


# ODE 
def derivatives(t, y):
    r, theta, vr, vtheta = y
    drdt = vr
    dthetadt = vtheta
    dvrdt = r * vtheta**2 - G * M / r**2
    dvthetadt = -2 * vr * vtheta / r
    return [drdt, dthetadt, dvrdt, dvthetadt]

# Generate training data with time as the input
def generate_training_data(num_samples=1000, t_span=(0, 10), t_eval_points=50):
    X_train = []  # Time values as input
    y_train = []  # Positions (r, theta) as output
    t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)

    for _ in range(num_samples):
        # Random initial conditions
        r0 = 1
        vr0 = 0  # Radial velocity should be zero for a circular orbit
        vtheta0 = np.sqrt(G * M / r0)  # Circular orbit condition
        theta0 = np.random.uniform(0, 2*np.pi)
        # Initial state vector
        y0 = [r0, theta0, vr0, vtheta0]

        # Solve the ODE
        sol = solve_ivp(derivatives, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

        # Append time values and corresponding positions as whole arrays
        X_train.append(t_eval)  # Append the whole t_eval array as one entry
        y_train.append(sol.y[:2, :].T)  # Append the (r, theta) array as one entry

    # Convert X_train and y_train to numpy arrays and reshape them as needed
    X_train = np.array(X_train).reshape(-1, 1)  # Reshape time values into a column vector
    y_train = np.array(y_train).reshape(-1, 2)  # Reshape (r, theta) values into 2D array

    return X_train, y_train

# Generate the training data
X_train, y_train = generate_training_data()

model = models.Sequential([
    layers.Input(shape=(1,)),  # time one input
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1024, activation='relu'),
    layers.Dense(2, activation='linear')  # Output shape matches r and theta
])
    
# Compile the model
model.compile(optimizer='adam', loss='mse')

custom_loss_history = CustomLossHistory()
history = model.fit(X_train, y_train, epochs=100, batch_size=5000, validation_split=0.2, callbacks=[custom_loss_history])

# Save the trained model in TensorFlow format
model.save("./trained_PINN_model_orbit_time_input_optimized", save_format='tf')

# Plot the loss history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss History')
plt.savefig("loss_history_plot_Orbit_Time_Input_Optimized")

# Plot loss history for r and theta
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(custom_loss_history.loss_r, label='Loss for r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss for r over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(custom_loss_history.loss_theta, label='Loss for theta')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss for theta over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig("loss_history_r_theta.png")
plt.show()

# Evaluate the model on new time values
def evaluate_model(model, t_span=(0, 5), t_eval_points=500):
    t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)

    # Predict positions (r, theta) for each time value
    y_pred = model.predict(t_eval.reshape(-1, 1))  # Reshape to (t_eval_points, 1)
    r_pred, theta_pred = y_pred[:, 0], y_pred[:, 1]

    return r_pred, theta_pred, t_eval

# Test the model with new time values
r_pred, theta_pred, t_eval = evaluate_model(model)

# Set up the figure for animation
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_ylim(0, 2)  # Set appropriate limits for r
line, = ax.plot([], [], 'o-', lw=2)

# Add a text box to display the time inside the plot (positioned at r=1.5, theta=0)
time_template = 'Time = {:.1f}s'  # Template for the time display
time_text = ax.text(0.0, 1.5, '', transform=ax.transData, fontsize=12, color='red')  # Inside the plot area

def init():
    line.set_data([], [])
    time_text.set_text('')  # Clear the time text
    return line, time_text

def update(frame):
    if frame < len(r_pred):
        line.set_data(theta_pred[:frame], r_pred[:frame])
        # Properly format and display the time value
        time_text.set_text(time_template.format(t_eval[frame]))  # Update the time display
    return line, time_text

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=30)

# Display the animation
plt.show()