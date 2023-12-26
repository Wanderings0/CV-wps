import wandb
import matplotlib.pyplot as plt

# Initialize wandb and replace 'your-entity/your-project/your-run-id' with your actual path
api = wandb.Api()

# Fetch the run from the API
run = api.run("your-entity/your-project/your-run-id")

# Retrieve the history for the run
history = run.history()
# Assuming 'train_acc' and 'test_acc' are the keys for the training and test accuracy
train_acc = history['train_acc']
test_acc = history['test_acc']

# Plotting
plt.figure(figsize=(12, 6))  # Bigger figure size

# Enhanced plot design
plt.plot(train_acc, label='Train Accuracy', color='blue', linewidth=2, marker='o', markersize=5, linestyle='--')
plt.plot(test_acc, label='Test Accuracy', color='red', linewidth=2, marker='s', markersize=5, linestyle='-.')

plt.xlabel('Epochs', fontsize=14)  # Larger font size
plt.ylabel('Accuracy', fontsize=14)
plt.title('Train vs Test Accuracy', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)  # Add gridlines

# Optionally, use a style context
with plt.style.context('ggplot'):
    plt.plot(train_acc, label='Train Accuracy', color='blue', linewidth=2, marker='o', markersize=5, linestyle='--')
    plt.plot(test_acc, label='Test Accuracy', color='red', linewidth=2, marker='s', markersize=5, linestyle='-.')

# Show or save the plot
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
# plt.show()
plt.savefig('accuracy.png', dpi=300)  # Save the plot as a PNG file

# Close the wandb run
# run.finish()